# Feature 02 — Same-model experiment

← [01 Hardware-aware optimizer](./01-hardware-aware-optimizer.md) · [Index](./README.md) · Next: [03 HTML report](./03-report-visualization.md) →

Priority: ⭐⭐⭐⭐⭐ · Phase 1 · Module: `experiment.py` · Command: `odg experiment` · Scripts: `benchmark/`

---

## Goal

Prove (or refute) OpenDynamicGGUF against the original model and against uniform GGUF baselines **on the same model, the same tokenizer, and the same lm-eval-harness config**. Quantization is the only independent variable.

Default test model: `google/functiongemma-270m-it`.

```text
                    SAME MODEL
                       │
             ┌─────────┴─────────┐
             │                   │
        Original model       OpenDynamicGGUF
          BF16/FP16             quantized
             │                   │
             └─────────┬─────────┘
                       │
                 SAME BENCHMARK
                       │
        ┌──────────────┼──────────────┐
        MMLU        GSM8K       HellaSwag
        ARC         TruthfulQA
```

```bash
./benchmark/run_all.sh
# or
odg experiment prepare
odg experiment run --all
odg experiment compare
```

Writes `benchmark/results/comparison.md` — quality, size, behavior, inference.

---

## Why it exists

A single GGUF score is not a claim. The claim is:

> At approximately the same compression level, OpenDynamicGGUF retains more
> benchmark performance than Q4_K_M.

That requires BF16 + uniform Q4_K_M / Q5_K_M / Q6_K + ODG, identical harness settings, and a size column. See [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the known GGUF-eval caveat ([issue 2887](https://github.com/EleutherAI/lm-evaluation-harness/issues/2887)).

---

## Pin (do not change between variants)

| Knob | Value |
|---|---|
| Model | `google/functiongemma-270m-it` |
| Tasks | `mmlu`, `gsm8k`, `hellaswag`, `arc_challenge`, `truthfulqa_mc2` |
| Shots | harness defaults (not a global `--num_fewshot`) |
| Seed | `0` |
| Batch size | `8` |
| Tokenizer | original HF tokenizer |
| Suite | `dev` (32 samples, laptop) or `paper` (full) |

Config lives in `benchmark/config.json`. A fingerprint is stored on every `result.json`; `compare` refuses to treat mismatched pins as comparable.

---

## Layout

```text
benchmark/
├── prepare.sh
├── run_bf16.sh
├── run_q4.sh
├── run_q5.sh
├── run_q6.sh
├── run_odg.sh
├── run_all.sh
├── compare.py
├── config.json
└── results/
```

---

## Done when

- [x] Old single-GGUF `odg benchmark` / `benchmark.py` removed
- [x] `./benchmark/run_all.sh` produces `results/comparison.md`
- [x] BF16, Q4_K_M, Q5_K_M, Q6_K, ODG share one eval command except `model_args`
- [x] Size / bytes-per-parameter sit next to quality
- [x] Missing lm-eval or GGUF is recorded as skipped — never faked

## Next

[Feature 03 — Interactive HTML report](./03-report-visualization.md)
