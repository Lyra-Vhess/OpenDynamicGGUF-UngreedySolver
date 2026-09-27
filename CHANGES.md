# Changes: DP-MCKP optimizer with column generation (`feat/dp-mckp-colgen`)

## Why

The Step-13 optimizer was a greedy downgrade loop: fast, but provably
suboptimal on non-concave cost steps (e.g. a group where Q4->Q3 is expensive
yet Q3->Q2 is nearly free — greedy commits to the expensive step and can
never reach the cheap one). It also swept the full `(group, quant)` probe
grid and optimized mean KLD only.

## What changed

- **Tail-KLD objective** (`kld.py`, new): `kld_tail_1pct` = mean KLD over the
  worst 1% of scored tokens via `numpy.partition` (no full sort); `kld_mean`
  still reported. Every sensitivity row now stores `kld_mean`,
  `kld_tail_1pct`, `n_tokens`. Proxy-mode rows use a provisional `mean x 8`
  tail until per-token dumps land (needs a minimal `--kld-output` patch to
  `llama-perplexity`, documented in `kld.py`).
- **DP MCKP solver** (`dp_mckp.py`, new): exact dynamic programming over the
  probed cost matrix, 1 MiB ceil bins (conservative: binned-feasible always
  fits the true budget). Full Pareto frontier falls out of one solve.
  Infeasible budgets fail loudly with minimum size + largest group.
- **Column generation** (`colgen.py`, new): floor-first probes, shadow-price
  pricing, monotone + Lipschitz bound model, termination certificate, and
  `exhaustive` mode for unconditional certification. Adaptive single-column
  batches (re-solve/re-price between probes) so early rounds don't degenerate
  into a full sweep.
- **CLI** (`cli.py`): `--optimizer {greedy,dp_mckp}` (default `dp_mckp`),
  `--kld-objective {tail_1pct,mean}`, `--certificate {bounded,exhaustive}`,
  `--lipschitz`, `--jobs`, `--pareto-ratios`. All existing flags preserved.
- **Recipe** (`optimizer.py`): additive `optimizer`, `kld_metric`,
  `cost_matrix`, `allocation`, `totals`, `certificate`, `pareto`,
  `discretization` sections. Greedy output is byte-identical to before.
- **Docs**: README DP-MCKP subsection; `TESTING.md` (this branch).

## Deprecated (not removed)

`--optimizer greedy` remains for A/B comparison. Removal awaits validation
on a real sweep per the spec. Default is now `dp_mckp`.

## Test summary

`python3 -m pytest tests/ -q` — 77 passed (21 new: 8 tail-KLD, 5 DP,
4 colgen, 4 optimize-integration). No new dependencies (`numpy` only).

## Deviations from Spec.md

- Shadow-price sign fixed: implemented as `(V[B-D] - V[B]) / D_bytes`
  (non-negative); spec text had the subtraction reversed.
- Pricing uses `extra = bytes[q] - bytes[floor]` (>= 0) instead of spec's
  `bytes[floor] - bytes[q]`, since the floor type is the smallest — same
  inequality, corrected orientation.
- Imatrix scales normalized to [0.2, 1.0] (0.2 floor) rather than [0, 1] so
  the least-sensitive group keeps a non-degenerate bound.
- `batch_size = 1` default (re-solve/re-price between probes); certificate
  semantics unchanged, probe count strictly lower in practice.
- Commits unsigned: the checkout's SSH signing key requires a passphrase
  and no agent is running. Push blocked: `origin` is upstream-https with no
  fork remote — create a fork and push `feat/dp-mckp-colgen` from there.
