# Testing the DP-MCKP optimizer

Target hardware: single 16 GB GPU, 128 GB RAM.

## Setup (no venv, `odg` not installed)

This checkout is not a virtualenv and the package is not installed, so
there is no `odg` command. Everything below works with the system Python
directly — all runtime dependencies (`numpy`, `rich`, `huggingface_hub`)
and `pytest` are already installed. Just `cd` to the repo root:

```
cd /home/lyra/AI/OpenCode/OpenDynamicGGUF
```

Then substitute `python3 cli.py` for every `odg` in this guide, e.g.
`odg optimize ...` becomes:

```
python3 cli.py optimize ...
```

(Optional: `pip install -e .` creates a real `odg` command from this
checkout. On this machine pip requires the `--break-system-packages`
flag. Not needed for any test below.)

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

Use the 270M test model with proxy sensitivity (no trial quants, no GPU).
All commands run from the repo root with `python3 cli.py` (see Setup):

```
python3 cli.py run --model functiongemma:latest --quant q4_k_m --no-ask --quiet
python3 cli.py optimize --budget-mb 180
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
python3 cli.py export --mode llama
python3 cli.py validate
```

`validate` runs on the held-out split only; the tail metric is computed on
the search split during probing.

## 2. Real Gemma E4B sweep (bounded mode)

```
python3 cli.py run --model <gemma-e4b-ref> --quant q4_k_m --no-ask --until sensitivity
python3 cli.py reband
python3 cli.py sensitivity --mode llama --force
python3 cli.py optimize --budget-ratio 0.72 --jobs 2
```

Notes:

- `--jobs` controls process-level probe parallelism (each probe is a
  `llama-quantize` + `llama-perplexity` subprocess pair). Start with 2 on a
  16 GB GPU: one trial quant resident plus the reference model must fit in
  VRAM together. Raise to 3–4 only if `nvidia-smi` shows headroom during a
  probe; lower to 1 at the first OOM (see troubleshooting).
- GPU offload is off by default (CPU everywhere). If your llama.cpp build
  has a GPU backend compiled in, add `--perplexity-args "-ngl 99"` to the
  `sensitivity` / `reference-logits` commands and `--imatrix-args "-ngl 99"`
  to `imatrix` — flags pass through verbatim, no rebuild needed. The flag
  spelling is identical on CUDA, Vulkan, Metal, and ROCm builds. Check
  `llama-perplexity --help` lists `-ngl` first; if it doesn't, your build
  is CPU-only and the flags will error.
- The group count for this class of model is ~25 (one `role@depth` group per
  role per depth band, plus global embedding/lm_head). Bounded mode probes
  ~|G| floor columns first, then only attractive columns — far fewer than
  the full |G| x |Q| sweep.
- `odg reband` (step 11b) re-cuts each role's layers into bands on
  measured importance cliffs (exact Fisher-Jenks segmentation over real
  `imatrix.gguf` per-channel stats, proxy fallback) instead of hard
  thirds — same band count, only boundaries move. `sensitivity` uses the
  rebanded catalog automatically when present and warns if it went stale.
- No patched binaries needed: the stock `llama-perplexity --kl-divergence`
  run prints a `99.0% KLD` percentile line, and that is the tail metric
  (the threshold above which the worst 1% of tokens sit; see `kld.py`).
  Proxy-mode rows carry no tail (`kld_tail_1pct` is null) and the DP tail
  objective refuses them with a hard error pointing at
  `sensitivity --mode llama` — there is no silent estimate anywhere.
- Step 12 in llama mode quantizes one trial GGUF per probed
  `(group, type)` and runs perplexity against the step-11 KL base on the
  search split. Trials land under the step directory (`trials/`).
  `--jobs N` parallelizes probes; binaries resolve from PATH or
  `--llama-quantize` / `--llama-perplexity`. A missing `99.0% KLD` line in
  any probe log is a hard error, not a skipped column.
- The probe grid is per-group, not global: each group is probed only at
  the types on its own candidate ladder (pins included). Above-ladder
  types the DP could never choose (e.g. Q8 for unpinned groups,
  everything below floor for pinned ones) are skipped before any GPU work
  — on the 270M pilot this cut 132 probes to 97. The step log reports the
  skipped count; a grid that covers nothing on some group's ladder is a
  hard error naming the group (widen `--probe-types`).

## 3. Exhaustive re-run (certified final recipe)

```
python3 cli.py optimize --budget-ratio 0.72 --certificate exhaustive --jobs 2 --force
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
python3 cli.py optimize --budget-ratio 0.72 --optimizer dp_mckp --force
python3 cli.py optimize --budget-ratio 0.72 --optimizer greedy --force
```

Recommended rematch (mean objective with automatic P99 guardrail — no
flag; the optimizer derives the cap from its own first pass):

```
python3 cli.py optimize --budget-mb 270 --kld-objective mean
python3 cli.py optimize --budget-mb 270 --optimizer greedy
```

then export each `recipe.tt` and measure both GGUFs with
`llama-perplexity -m CANDIDATE -f heldout.txt --kl-divergence
--kl-divergence-base <step-11 logits-heldout.bin>`.

Compare `totals.kld_mean` / `totals.kld_tail` (DP) vs
`estimate.predicted_mean_delta_kld` (greedy), and diff the two `recipe.tt`
files. DP is guaranteed optimal over the probed cost matrix, but note: the
*tail* objective minimizes a sum of per-group P99s, which is not a
whole-model percentile and can fairly lose to greedy on measured mean, P99,
and same-top. The mean objective plus the automatic guardrail (pass 1 solves
mean-only, pass 2 re-solves mean with every group's P99 capped at the worst
P99 of the pass-1 allocation) is the recommended comparison — there is no
cap to pick by hand.

## 6. Troubleshooting

**VRAM exhausted during probes.** Lower `--jobs` to 1. If a single probe
still OOMs, the reference model plus one trial quant exceeds 16 GB: run
`sensitivity --mode proxy` for plumbing, and move real `llama` probes to a
bigger machine. DP/colgen logic is unchanged; only the cost-matrix values
differ.

**DP reports infeasible.** The error states the minimum achievable size and
the largest minimum-size group. Causes: budget below the pin floor
(embeddings/lm_head at Q8, attn_v at Q5 minimum) — now quoted *with* the
1.09 size margin and 1 MiB per-group ceil waste, so pad the budget ~10–15%
above the raw estimate. Fixes in order: raise `--budget-mb`, or re-run
with `--no-pins` (accepts quality risk on pinned roles — not recommended
for release).

**Certificate never terminates (non-empty `attractive_at_termination`).**
The shadow price `λ` is oscillating on discretization noise: widen the
finite-difference window (currently 16 MiB bins in `colgen.py`) or smooth
`λ` across iterations. Do not lower `lipschitz_L` to force termination —
that invalidates the bound. If oscillation persists, run `--certificate
exhaustive` for an unconditional result.

**Tail metric looks noisy across runs.** Expected: the worst-1% P99 is
noisier than the overall mean. The default mean objective with the automatic
guardrail already handles this (the cap constrains per-group worst-1% while
the mean decides); cross-check `kld_mean` against the `guardrail` block in
the recipe, and if they disagree sharply on which allocation wins, enlarge
the search split before trusting the tail.
