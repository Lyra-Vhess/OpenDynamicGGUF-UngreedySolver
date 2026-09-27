# Changes: DP-MCKP optimizer with column generation (`feat/dp-mckp-colgen`)

## Why

The Step-13 optimizer was a greedy downgrade loop: fast, but provably
suboptimal on non-concave cost steps (e.g. a group where Q4->Q3 is expensive
yet Q3->Q2 is nearly free — greedy commits to the expensive step and can
never reach the cheap one). It also swept the full `(group, quant)` probe
grid and optimized mean KLD only.

## What changed

- **Tail-KLD objective** (`kld.py`, new): `kld_tail_1pct` = the `99.0% KLD`
  percentile line of a stock `llama-perplexity --kl-divergence` run — the
  threshold above which the worst 1% of tokens sit. No upstream C++ patch:
  the spec's exact top-1% mean needs a per-token dump the stock binary does
  not emit, and P99 targets the same tail conservatively (top-1% mean is
  always >= P99). `kld_mean` still reported. Every sensitivity row stores
  `kld_mean`, `kld_tail_1pct`, `n_tokens`. Proxy-mode rows carry
  `kld_tail_1pct = None` (no estimate); the DP tail objective refuses them
  with a hard error. A missing `99.0% KLD` line is likewise a hard error,
  never a silent fallback.
- **Measured per-group probes** (`llama_probe.py`, new; `sensitivity.py`):
  `--mode llama` on step 12 now really measures: one trial quantize per
  probed `(group, type)` (single-group `--tensor-type-file` override,
  all-baseline anchor for deltas), perplexity against the step-11 KL base
  on the search split, parsed via `kld.py`. Thread-pool parallelism via
  `--jobs`; binaries from PATH or `--llama-quantize`/`--llama-perplexity`.
  Verified on a 2-group x 2-type smoke test with monotonic, tail-above-mean
  results.
- **DP MCKP solver** (`dp_mckp.py`, new): exact dynamic programming over the
  probed cost matrix, 1 MiB ceil bins (conservative: binned-feasible always
  fits the true budget). Full Pareto frontier falls out of one solve.
  Infeasible budgets fail loudly with minimum size + largest group.
- **P99 guardrail** (`optimizer.py`, `--tail-cap TAU`): measured-rematch
  finding — a *summed* P99 is not a whole-model percentile and misranks
  allocations, while mean KLD is approximately additive. So the recommended
  shape is mean objective plus a per-group worst-1% cap: cap-violating
  columns are deleted pre-DP (needs measured tails; hard-errors on proxy
  rows; loud error if a group is emptied). DP, Pareto, and certificate run
  unchanged on the restricted problem.
- **Size-estimate margin** (`optimizer.py`, hard-coded `SIZE_ESTIMATE_MARGIN`
  = 1.09): real exports run ~8–9% over estimates (single 270M-model
  calibration; replace with an empirically derived value or a better
  estimator when available). Inherited by both optimizers and all budget
  ratios; visible in recipe/manifest.
- **Column generation** (`colgen.py`, new): floor-first probes, shadow-price
  pricing, monotone + Lipschitz bound model, termination certificate, and
  `exhaustive` mode for unconditional certification. Adaptive single-column
  batches (re-solve/re-price between probes) so early rounds don't degenerate
  into a full sweep.
- **CLI** (`cli.py`): `--optimizer {greedy,dp_mckp}` (default `dp_mckp`),
  `--kld-objective {tail_1pct,mean}`, `--tail-cap`,
  `--certificate {bounded,exhaustive}`,
  `--lipschitz`, `--jobs`, `--pareto-ratios`, `--probe-types`.
  All existing flags preserved.
- **Recipe** (`optimizer.py`): additive `optimizer`, `kld_metric`,
  `cost_matrix`, `allocation`, `totals`, `certificate`, `pareto`,
  `discretization` sections. Greedy output is byte-identical to before.
- **Docs**: README DP-MCKP subsection; `TESTING.md` (this branch).

## Deprecated (not removed)

`--optimizer greedy` remains for A/B comparison. Removal awaits validation
on a real sweep per the spec. Default is now `dp_mckp`.

## Test summary

`python3 -m pytest tests/ -q` — 84 passed (4 new: guardrail optimality vs
brute force, cap-too-tight error, cap-vs-proxy hard error, margin
inflation). No new dependencies (`numpy`
only; the log parser is stdlib).

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
