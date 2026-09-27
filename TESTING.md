# Testing the DP-MCKP optimizer

Target hardware: single 16 GB GPU, 128 GB RAM. All unit tests run anywhere
with `python3 -m pytest tests/ -q` (no GPU, no models, no llama.cpp).

## 0. Unit tests (no hardware needed)

```
python3 -m pytest tests/ -q
```

Expect 77 passed. The DP-specific tests:

- `tests/test_tail_kld.py` — tail-1% metric vs hand-computed values.
- `tests/test_dp_mckp.py` — DP equals brute force; greedy provably fails the
  non-concave toy; Pareto monotone; infeasibility guard.
- `tests/test_colgen.py` — column generation equals brute force on 81
  combos; negative test with an invalid (too-small) Lipschitz bound.
- `tests/test_optimize_dp.py` — end-to-end optimize step with DP default,
  greedy unchanged, flag plumbing.

## 1. Small-scale end-to-end (VRAM-light)

Use the 270M test model with proxy sensitivity (no trial quants, no GPU):

```
odg run --model functiongemma:latest --quant q4_k_m --no-ask --quiet
odg optimize --budget-mb 180
```

Expected runtime: seconds for the optimize step (DP over ~25 groups x ~3 GiB
of 1 MiB bins is milliseconds; proxy probes are arithmetic). Artifacts land
in the run's `steps/13_optimize/` directory:

- `recipe.yaml` — primary recipe, now with `optimizer: dp_mckp`,
  `kld_metric`, `cost_matrix`, `allocation`, `totals`, `certificate`,
  `pareto`, `discretization` sections.
- `recipe.tt` — tensor-type file for `llama-quantize --tensor-type-file`.
- `pareto/pareto-*.yaml` — one recipe per budget ratio.
- `optimize_manifest.json` — machine-readable summary incl. certificate.

Then export and validate as usual:

```
odg export --mode llama
odg validate
```

`validate` runs on the held-out split only; the tail metric is computed on
the search split during probing.

## 2. Real Gemma E4B sweep (bounded mode)

```
odg run --model <gemma-e4b-ref> --quant q4_k_m --no-ask --until sensitivity
odg sensitivity --mode llama --force
odg optimize --budget-ratio 0.72 --jobs 2
```

Notes:

- `--jobs` controls process-level probe parallelism (each probe is a
  `llama-quantize` + `llama-perplexity` subprocess pair). Start with 2 on a
  16 GB GPU: one trial quant resident plus the reference model must fit in
  VRAM together. Raise to 3–4 only if `nvidia-smi` shows headroom during a
  probe; lower to 1 at the first OOM (see troubleshooting).
- The group count for this class of model is ~25 (one `role@depth` group per
  role per depth band, plus global embedding/lm_head). Bounded mode probes
  ~|G| floor columns first, then only attractive columns — far fewer than
  the full |G| x |Q| sweep.
- Per-token KLD requires a `llama-perplexity` build with the minimal
  `--kld-output <file>` patch (dumps the internal `kld_values` array; see
  `kld.py` header). Without it, rows carry the provisional proxy tail
  (`mean x 8`) and `n_tokens` is null — good for plumbing, not for release.

## 3. Exhaustive re-run (certified final recipe)

```
odg optimize --budget-ratio 0.72 --certificate exhaustive --jobs 2 --force
```

Probes every `(group, quant)` column; the emitted certificate is then
unconditional. Use this for the published recipe; use `bounded` for
iteration.

## 4. Reading the certificate

In `recipe.yaml`, find the `certificate:` block:

```
certificate:
  mode: bounded            # or "exhaustive"
  bound_model: { monotonic: true, lipschitz_L: <float>, ... }
  probed_columns: <int>
  excluded_columns: <int>
  shadow_price_lambda: <float>
  attractive_at_termination: []   # must be empty
```

Checklist:

1. `attractive_at_termination` is `[]`. Non-empty means the loop hit its
   iteration cap without certifying — do not ship that recipe.
2. `mode: bounded` is conditional on `bound_model`: monotonicity plus the
   stated `lipschitz_L`. `mode: exhaustive` with `excluded_columns: 0` is
   unconditional.
3. `probed_columns + excluded_columns` equals the full column count
   (|G| x |Q| over probed groups). Every column is accounted for.
4. `shadow_price_lambda` is the marginal tail-KLD per byte at the solution;
   near-zero means budget headroom (a smaller budget may give the same
   quality).

## 5. Comparing against the greedy baseline

Same budget, same sensitivity table, both optimizers:

```
odg optimize --budget-ratio 0.72 --optimizer dp_mckp --force
odg optimize --budget-ratio 0.72 --optimizer greedy --force
```

Compare `totals.kld_tail` (DP) vs `estimate.predicted_mean_delta_kld`
(greedy), and diff the two `recipe.tt` files. DP is guaranteed optimal over
the probed cost matrix; any gap favoring greedy indicates a stale
sensitivity table (re-run `sensitivity --force`), not a DP bug.

## 6. Troubleshooting

**VRAM exhausted during probes.** Lower `--jobs` to 1. If a single probe
still OOMs, the reference model plus one trial quant exceeds 16 GB: run
`sensitivity --mode proxy` for plumbing, and move real `llama` probes to a
bigger machine. DP/colgen logic is unchanged; only the cost-matrix values
differ.

**DP reports infeasible.** The error states the minimum achievable size and
the largest minimum-size group. Causes: budget below the pin floor
(embeddings/lm_head at Q8, attn_v at Q5 minimum). Fixes in order: raise
`--budget-mb`, or re-run with `--no-pins` (accepts quality risk on pinned
roles — not recommended for release).

**Certificate never terminates (non-empty `attractive_at_termination`).**
The shadow price `λ` is oscillating on discretization noise: widen the
finite-difference window (currently 16 MiB bins in `colgen.py`) or smooth
`λ` across iterations. Do not lower `lipschitz_L` to force termination —
that invalidates the bound. If oscillation persists, run `--certificate
exhaustive` for an unconditional result.

**Tail metric looks noisy across runs.** Expected: the worst-1% mean is
noisier than the overall mean. Keep `--kld-objective tail_1pct` for the
optimizer but cross-check `kld_mean` in the recipe; if they disagree
sharply on which allocation wins, enlarge the search split before trusting
the tail.
