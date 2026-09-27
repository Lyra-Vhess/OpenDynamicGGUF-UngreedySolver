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
  results. The grid is per-group: each group is probed only at the types on
  its own pins-aware ladder, so above-ladder types the DP could never choose
  (Q8 for unpinned groups, below-floor types for pinned ones) are skipped
  before any GPU work — 132 probes to 97 on the 270M pilot. A grid covering
  nothing on some group's ladder is a hard error naming the group.
- **DP MCKP solver** (`dp_mckp.py`, new): exact dynamic programming over the
  probed cost matrix, 256 KiB ceil bins (conservative: binned-feasible always
  fits the true budget). Full Pareto frontier falls out of one solve.
  Infeasible budgets fail loudly with minimum size + largest group.
- **P99 guardrail, automatic** (`optimizer.py`, no flag): measured-rematch
  finding — a *summed* P99 is not a whole-model percentile and misranks
  allocations, while mean KLD is approximately additive. So under the mean
  objective the optimizer runs two passes: pass 1 solves mean-only, takes
  the worst per-group P99 of that allocation as T*, and pass 2 re-solves
  mean subject to every group staying at or below T* (provably feasible;
  needs measured tails, hard-errors on proxy rows). T*, both pass means,
  and the removal count land in recipe/manifest. DP, Pareto, and
  certificate run unchanged on the restricted problem.
- **Measured sizes, no margin** (`llama_probe.py`, `optimizer.py`):
  probed columns use exact group bytes read from trial-file GGUF metadata
  (`bytes_measured`, audited per entry); never-probed columns fall back to
  bytes-per-element estimates flagged `measured: false`. The old 1.09
  constant is deleted (defaults are a neutral 1.0). A dual-threshold sanity
  check (absolute dev > 256 KiB AND relative dev > 15%) aborts the optimize
  naming the corrupt column instead of solving on it.
- **Column generation** (`colgen.py`, new): floor-first probes, shadow-price
  pricing, monotone + Lipschitz bound model, termination certificate, and
  `exhaustive` mode for unconditional certification. Adaptive single-column
  batches (re-solve/re-price between probes) so early rounds don't degenerate
  into a full sweep.
- **CLI** (`cli.py`): `--optimizer {greedy,dp_mckp}` (default `dp_mckp`),
  `--kld-objective {tail_1pct,mean}` (default `mean`),
  `--certificate {bounded,exhaustive}`,
  `--lipschitz`, `--jobs`, `--pareto-ratios`, `--probe-types`,
  `--perplexity-args` (sensitivity, reference-logits) and `--imatrix-args`
  (imatrix): verbatim passthrough to llama.cpp binaries, e.g. `"-ngl 99"`
  for GPU offload on any backend (CUDA/Vulkan/Metal/ROCm) with no rebuild.
  All existing flags preserved.
- **Reband** (`reband.py`, new; step 11b `odg reband`, in `odg run` after
  imatrix): replaces thirds banding with exact Fisher-Jenks segmentation
  per role over real `imatrix.gguf` per-channel stats (proxy fallback);
  roles with no band structure keep thirds (elbow guard); unscored layers
  join the nearest band explicitly. Same group count, boundaries move.
- **Proxy inf fix** (`imatrix.py`): non-finite raw scores (e.g. rope_freqs
  spectral blowup → inf, which had zeroed the whole file via /inf) are
  excluded and audited instead of normalizing everything to 0.
- **Recipe** (`optimizer.py`): additive `optimizer`, `kld_metric`,
  `cost_matrix`, `allocation`, `totals`, `certificate`, `pareto`,
  `discretization` sections. Greedy output is byte-identical to before.
- **Docs**: README DP-MCKP subsection; `TESTING.md` (this branch).
- **Fixed groups** (`--fixed-groups`, `optimizer.py`, `llama_probe.py`):
  groups that must stay at source precision (tensors llama-quantize
  cannot quantize, e.g. arch-unknown 2-D projections) are excluded from
  candidates and the recipe; their catalog bytes are subtracted from every
  budget before solving and added back to every total, so `--budget-mb`
  keeps meaning actual file size. Probes now assert every non-flat group
  tensor actually took the probe type in the trial file — the old silent
  no-match (bogus zero-delta rows) is a hard error instead. Stale
  trial-*.gguf files are cleared when probing starts.

## Deprecated (not removed)

`--optimizer greedy` remains for A/B comparison. Removal awaits validation
on a real sweep per the spec. Default is now `dp_mckp`.

## Test summary

`python3 -m pytest tests/ -q` — 109 passed (7 new: probe-effect assert
×3, fixed-group DP/greedy accounting ×4). No new dependencies (`numpy`
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
