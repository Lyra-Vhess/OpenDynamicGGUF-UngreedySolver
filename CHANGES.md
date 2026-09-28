# Changes: DP-MCKP optimizer with column generation (`feat/dp-mckp-colgen`)

## E4B lazy-pipeline rerun: gap reversed by the pipeline itself (2026-09-28)

- **Step 12 --force** (pid 767486, `/tmp/lazy_sens4.log`): 193 rows, 84
  pricing rounds, 7 bound-excluded, lambda=0.0, 7 `pin_high` hints (late
  layers + other@early/late). Sidecar harvest recovered 49/52 orphan
  trials (~287 GB freed, 2 partials remeasured); disk held 58–59%
  throughout. 1 orphan gguf remains for next-run harvest.
- **Reference-budget lesson**: the loose all-Q6×1.2 reference (7059 MiB)
  put lambda≈0, so pricing measured near-exhaustively (193/200) — sound
  (exclusions transfer down) but no probe savings. Fix, same commit:
  `--budget-mb`/`--budget-ratio` added to the sensitivity parser;
  `cmd_sensitivity` computes the intended solve bytes and passes them as
  `pricing_budget_bytes` (loose reference stays as API fallback);
  sensitivity stale-check gains `budget_mb`/`budget_ratio` keys.
- **Step 13** (`/tmp/lazy_opt5.log`; first attempt failed correctly — the
  operator forgot `--fixed-groups other@global`, DP took a proxy-Q2 column
  and the guardrail refused it): allocation Q5×10/Q8×5/Q4×5/Q6×4/Q2×1,
  **embedding@global→Q4_K on measured data**, predicted mean −0.00163,
  guardrail pass1→pass2 (−0.00211→−0.00163, T*=0.0328), bounded cert 119
  probed/43 excluded, 4881 MiB ≤ 4885 budget. Export matched prediction
  within 2.4 KB (5118550976 B).
- **Heldout Tier-1** (`/tmp/tier1_lazy.log`): mean 0.00633, P99 0.101,
  same-top 98.3% — beats the manual hand-export (0.0091/0.148/98.0) and XL
  (0.0178/0.251/97.2) at smaller size. The pipeline now reproduces on its
  own what took manual intervention in the postscript below.
- Follow-ups: BF16 stays off the ladder pending build verification;
  IQ4_XS still not on the ladder.
- **Measured Tier-1 wired** (`validate.py: _tier1_llama`, `cli.py`): step
  15 runs `llama-perplexity --kl-divergence` on the candidate against
  `logits-heldout.bin` whenever heldout assets exist (mode auto prefers
  measured, falls back to proxy; mode llama hard-errors naming what's
  missing). Gates are v1, E4B-calibrated (mean ≤ 0.05, P99 ≤ 0.50,
  top1 ≥ 0.90). New flags `--llama-perplexity` / `--perplexity-args` on
  `odg validate` (recorded in input.json + stale-check); `run`/`fit`
  already thread `--perplexity-args` through the shared namespace.

## Ladder reform: role pins gutted, F32/F16 ceiling, lazy probing wired (2026-09-28)

- **Role pins deleted** (`optimizer.py`, `sensitivity.py`, `report.py`,
  `cli.py`): `DEFAULT_PINS` (embedding/lm_head→Q8, attn_v→Q5) is gone, with
  it the `report.py` mirror, the `use_pins`/`pins` parameters, and the
  `--no-pins` flag on `optimize`/`run`/`fit` (passing it is now an argparse
  error). The 17x E4B gap postscript below showed the embedding pin cost
  1.1 GB for ~zero ΔKLD with no measured justification — heuristic floors
  are not coming back. Kept, because they are measured or explicit:
  `pin_high` hints (Q4 probe ΔKLD > 0.04 → Q5 floor), `--fixed-groups`,
  kept non-quantizable norms, and the validate feedback constraint.
- **Uniform ladder + F32/F16 ceiling** (`optimizer.py: LADDER`): every group
  shares one ladder, `F32 → F16 → Q8_0 → … → Q2_K`. The ladder is the
  candidate universe, not a bias: DP, pricing, and bounds treat every rung
  identically. Proxy multipliers added (`F16 ≈ 0.03`, `F32 ≈ 0.01`);
  `top_token_agree` gained its missing upper clamp (negative ΔKLD vs the
  anchor used to yield >100% agreement). Greedy stays K-ladder-scoped
  (downgrade-only walk; ceiling is DP-only) as the frozen A/B baseline.
- **Lazy probing actually wired** (`sensitivity.py: probe_groups_lazy`,
  `colgen.py: batch_probe_fn`): step 12 in llama mode now does what the
  colgen docstring always claimed — anchor + floor columns first, then
  priced rounds where `probe_fn` is a real `measure_column` call, stopping
  on the certificate. Previously step 12 measured the grid exhaustively and
  step-13 pricing was a row lookup, so pricing saved zero GPU probes. The
  pricing reference budget is all-Q6 × 1.2 (loose is the safe direction:
  excluded-at-loose stays excluded at any tighter real budget; 1.2 covers
  the loosest shipped format, q8_0 at 1.15). `--jobs` sizes the priced
  batch (probed in parallel; mandatory floor probes go out as one
  parallel batch — no laziness lost); resume is sidecar-based
  (`trials/probed.jsonl`: one line per success — KL metrics + byte counts,
  everything downstream of a trial file), so disk stays at ~1 in-flight
  trial instead of accumulating GBs per probe; orphan trial pairs from a
  killed run are harvested into the sidecar on start (their `.gguf`s
  deleted after parsing, logs kept); `--certificate exhaustive` keeps
  full-universe measurement. Pricing scales come from the run's
  `imatrix.gguf`
  (per-group means via `reband.real_imatrix_scores`, proxy-JSON fallback,
  neutral otherwise — neutral only widens bounds). New step-12 flags:
  `--kld-objective`, `--certificate`, `--lipschitz` (also threaded through
  `run`/`fit` stale-check).
- **Guardrail drops unmeasured columns** (`optimizer.py`): cap filtering
  removes columns with no measured P99 (counted as `cap_dropped_unmeasured`
  in recipe/manifest) instead of erroring — a column with no P99 claim
  cannot pass the guardrail, and the certificate covers the exclusion.
  Fully-unmeasured tables still fail loudly (`removes every candidate`).
- Tests: 144 green (was 126). New `tests/test_ladder_reform.py` (uniform
  ladder, pin_high survival, negative-delta ceiling win, agreement clamp,
  priced ceiling skip, sidecar resume + roundtrip/corruption rules,
  harvest, imatrix aggregation) plus `tests/test_validate_tier1.py`
  (measured Tier-1 pass/fail/errors, mode branches, CLI flags +
  stale-check key) plus
  `test_colgen.py` batch-parity tests (floors go out as one parallel batch —
  mandatory, so no laziness lost); reworked pin asserts in `test_report.py`,
  `test_tail_kld.py`, `test_optimize_dp.py`, `test_run_passthrough.py`
  (incl. `--no-pins` rejection on all four parsers).
- Follow-ups, not done here: GPU-verify an F32/F16 trial probe + export
  line against the bundled llama.cpp build; BF16 stays off the ladder until
  that verification passes for it.

## Why

The Step-13 optimizer was a greedy downgrade loop: fast, but provably
suboptimal on non-concave cost steps (e.g. a group where Q4->Q3 is expensive
yet Q3->Q2 is nearly free — greedy commits to the expensive step and can
never reach the cheap one). It also swept the full `(group, quant)` probe
grid and optimized mean KLD only.

## What changed

- **Mean-KLD objective with P99 guardrail** (`kld.py`, new): every
  sensitivity row stores `kld_mean` plus `kld_tail_1pct` — the `99.0% KLD`
  percentile line of a stock `llama-perplexity --kl-divergence` run (no
  upstream C++ patch: the exact top-1% mean needs a per-token dump the stock
  binary does not emit, and P99 targets the same tail conservatively).
  The optimizer **minimizes the mean** (additive across groups), while each
  group's P99 acts as an automatic guardrail: pass 1 solves mean-only, pass 2
  re-solves mean subject to every group staying within the worst pass-1 P99
  (no hand-set cap; `--kld-objective tail_1pct` remains for experiments).
  Proxy-mode rows carry `kld_tail_1pct = None` (no estimate); the tail
  objective and the guardrail refuse them with a hard error. A missing
  `99.0% KLD` line is likewise a hard error, never a silent fallback.
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
   before any GPU work — 132 probes to 97 on the 270M pilot. The default
   grid is the union of the profile grid with every type on any qualifying
   group's ladder, so pinned above-baseline types (Q8) are always covered
   without manual `--probe-types`. A grid covering
   nothing on some group's ladder is a hard error naming the group.
- **Imatrix-fed trials** (`cli.py`, `sensitivity.py`): the sensitivity step
   resolves the run's step-10 `imatrix.gguf` and passes it into every probe
   trial (previously trials silently ran without imatrix, misranking columns
   — wiring it in nearly halved the 5126 MB rematch loss). The resolved path
   (or null when step 10 produced no file) is recorded in the step input.
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
   `--fixed-groups` (sensitivity + optimize),
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
- **Docs**: README DP-MCKP subsection (`TESTING.md` kept local-only,
  gitignored like `Spec.md`).
- **Fixed groups** (`--fixed-groups`, `optimizer.py`, `llama_probe.py`):
  groups that must stay at source precision (tensors llama-quantize
  cannot quantize, e.g. arch-unknown 2-D projections) are excluded from
  candidates and the recipe; their catalog bytes are subtracted from every
  budget before solving and added back to every total, so `--budget-mb`
  keeps meaning actual file size. Probes now assert every non-flat group
  tensor actually took the probe type in the trial file — the old silent
  no-match (bogus zero-delta rows) is a hard error instead. Stale
   trial-*.gguf files are cleared when probing starts. `--fixed-groups` also
   exists on `sensitivity`: fixed groups are skipped at probe time (recorded
   as `fixed_skipped`) instead of measured into bogus rows.
- **Whole-file budget accounting** (`optimizer.py`, `cli.py`): `--budget-mb`
  is actual file bytes. Kept non-quantizable catalog bytes are counted into
  every total (previously omitted); file overhead (freeze GGUF `data_offset`
  + 4 KiB safety for export-added KV, measured in `cmd_optimize`, 0 with a
  warning on fallback) is subtracted from the budget pre-solve and added
  back everywhere; recipe/manifest/`OptimizeResult` carry a `budget` block
  and the new fields. Over-budget floors fail loudly itemizing the parts.
- **Single-command flag parity** (`cli.py`): `odg run` (and `odg fit`)
  accept every result-affecting step flag — `--mode`, `--target-tokens`,
  `--seed`, `--max-docs`, `--freeze-mode`, `--convert-script`,
  `--require-bf16`, `--chunks`, `--imatrix-args`, `--perplexity-args`,
  `--bands-per-role`, `--jobs`, `--probe-types`, `--fixed-groups`,
  `--optimizer`, `--kld-objective`, `--certificate`, `--lipschitz`,
  `--pareto-ratios`, `--no-pins`, `--budget-mb`, `--budget-ratio`
  (`run` only; `fit` keeps its hardware-derived budget), `--export-mode`,
  `--base-type`, `--validate-mode`, `--strict`, `--only-quantizable` —
  and thread each into its step instead of running every step on hardcoded
  defaults. Freeze/export/validate keep their own mode flags (their mode
  vocabularies differ from the global probe `--mode`).
- **Stale-checkpoint warn-and-confirm** (`cli.py`): when a checkpointed
  step's recorded `input.json` disagrees with the pipeline flags on a
  result-affecting key, `odg run`/`odg fit` warn itemizing the diffs and ask
  whether to re-run that step (default no). Non-interactive sessions
  (`--no-ask` or no TTY) warn and keep the checkpoint; `--force` re-runs
  without asking. `jobs` is parallelism-only and never triggers a prompt;
  keys absent from older checkpoints are ignored rather than false-positive.

## Deprecated (not removed)

`--optimizer greedy` remains for A/B comparison. Removal awaits validation
on a real sweep per the spec. Default is now `dp_mckp`.

## Test summary

`python3 -m pytest tests/ -q` — 134 passed (ladder reform ×8: uniform
ladder, pin_high survival, negative-delta ceiling win, agreement clamp,
priced ceiling skip, resume; run/fit flag parity + stale-confirm ×13
incl. --no-pins rejection; probe-effect assert ×3; fixed-group DP/greedy
accounting ×4; kept/overhead accounting ×2; probe-time fixed skip ×3). No
new dependencies (`numpy` only; the log parser is stdlib).

## Postscript (2026-09-28): the 17x gap is closed — it was the embedding pin

Size-matched E4B rematch stood at ours 0.29 vs third-party Q4_K_XL 0.017.
Per-tensor GGUF-header diff showed XL spends its budget nothing like us:
190 tensors ours-Q2→theirs-Q4 (+591 MB), 19→Q6 (+234), 82→F32 (+188),
24→Q5 (+148), 15→IQ4_XS (+56) — funded entirely by 2 tensors
ours-Q8→theirs-Q5 (**−1248 MB**: token_embd + per_layer_token_embd).
Byte-exact payload sums (4870 vs 4850 MB) confirm the trade balances.

Solver exonerated (DP optimum is exact; all-Q4-class over our matrix
predicts 0.0157 ≈ XL's 0.0178 — the matrix is honest), guardrail
non-binding (pass 1 == pass 2, T* removed nothing), λ=0.0 explained (DP
solution locally flat within the 16 MiB shadow-price window, not a bug).

Manual embedding probes (same Q6 anchor/imatrix/search split as step 12):
Q6 0.00295, **Q5 0.00276** — the Q8 pin cost 1.1 GB for ~zero quality.
CPU re-solve with the two measured columns (ladder extended, everything
else identical) predicts mean **0.0054** at 4885 MiB; hand-exported
candidate measures on heldout Tier-1: **mean 0.0091, P99 0.148,
same-top 98.0%, PPL ratio 1.002 at 4885 MiB** vs XL's 0.0178 / 0.251 /
97.2% / 1.015 at 5126 MiB. Gap reversed at 241 MB smaller file.

Follow-ups, not done here: ~~productize an override for pinned-group
ladders (today only the manual path can probe embedding@Q5 — per-group
grid filtering intersects even explicit `--probe-types` with the
pins-only ladder)~~ DONE 2026-09-28 by the ladder reform above (pins
gutted, uniform ladder, lazy pricing — no override needed because there
is nothing left to override); consider IQ4_XS on the ladder; the 82 small
`other@*` tensors XL keeps at F32 (+188 MB) are now reachable via the
F32/F16 ceiling (pricing measures them iff attractive). Experiment
artifacts (allocation, recipe.tt, quantize command, Tier-1 logs) are
outside the repo.

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
