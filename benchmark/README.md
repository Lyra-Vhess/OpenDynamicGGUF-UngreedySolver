# Same-model experiment

This directory is the **controlled comparison** for OpenDynamicGGUF.

The claim is not “this GGUF scores well on MMLU.” The claim is:

> On the **same model**, with the **same tokenizer** and the **same test questions**,
> OpenDynamicGGUF retains more task quality than a uniform GGUF of similar size.

Quantization is the only independent variable.

```text
                 same weights
                      │
        ┌─────────────┼─────────────┐
        │             │             │
     Source        Q4_K_M      OpenDynamicGGUF
     (local)     Q5_K_M / Q6_K     (recipe)
        │             │             │
        └─────────────┼─────────────┘
                      │
              same test splits
                      │
           MMLU · GSM8K · HellaSwag · ARC-Challenge
```

Default subject: local Ollama **`functiongemma:latest`** (FunctionGemma 270M, 268,098,176 parameters). Upstream checkpoint for a later BF16 run: [`google/functiongemma-270m-it`](https://huggingface.co/google/functiongemma-270m-it).

---

## What stays pinned

| Knob | Value |
|---|---|
| Model | One checkpoint, reused for every variant |
| Tasks | MMLU, GSM8K, HellaSwag (labeled val), ARC-Challenge |
| Splits | **Test only** (HellaSwag: original val — the public test file has no labels) |
| Subsets | First *N* rows of the published file order (same items on every GGUF) |
| Shots | MMLU 5-shot (per-subject `dev` CSV) · GSM8K 8-shot (start of official `train.jsonl`) · HellaSwag / ARC 0-shot |
| Suites | `dev` (default) and `release` — see below |

Training data is never scored. Files are read in their **published formats** (Hendrycks CSV, OpenAI JSONL, HellaSwag JSONL, AI2 JSONL). They are not rewritten into a custom schema first.

Suite caps live in [`../evaluation/config.json`](../evaluation/config.json). Do not change them between variants of the same table.

---

## Variants

| Alias | File under `benchmark/models/` | Role |
|---|---|---|
| **Source** | `functiongemma-270m-bf16.gguf` | Local original GGUF (often Ollama **Q8_0**, not BF16) |
| **Q4_K_M** | `functiongemma-270m-q4_k_m.gguf` | Uniform llama.cpp baseline |
| **Q5_K_M** | `functiongemma-270m-q5_k_m.gguf` | Uniform baseline |
| **Q6_K** | `functiongemma-270m-q6_k.gguf` | Uniform baseline |
| **OpenDynamicGGUF** | `functiongemma-270m-odg.gguf` | Per-tensor recipe from `odg run` |

`models/original.gguf`, `models/q4_k_m.gguf`, and `models/opendynamic.gguf` are aliases of those files for `python benchmark.py`.

A better score at a **much larger** file than Q4_K_M is not an interesting quantization result. The comparison that matters is quality **at similar size**.

---

## Two suites

| Suite | MMLU | GSM8K | HellaSwag | ARC-Challenge | Use |
|---|---:|---:|---:|---:|---|
| **`dev`** (default) | 500 | 200 | 500 | 200 | Every change to the quantizer (~1,400 items) |
| **`release`** | full ~14,042 | full 1,319 | full val ~10,042 | full 1,172 | Published comparison (~26K items) |

Never mix a `dev` table with a `release` table. `--limit N` overrides every task to *N* items; `--limit 0` forces the full split.

FunctionGemma 270M is a **function-calling** specialist. Absolute MMLU / GSM8K will be low. Report **retention vs the source GGUF**, not SOTA.

---

## Layout

```text
benchmark/
├── README.md                 ← this file
├── config.json               ← experiment pin (variants, seed)
├── prepare.sh                ← odg experiment prepare
├── run_all.sh                ← prepare + run + compare (lm-eval path, if installed)
├── models/                   ← GGUFs (gitignored)
│   ├── functiongemma-270m-bf16.gguf
│   ├── functiongemma-270m-q4_k_m.gguf
│   ├── functiongemma-270m-q5_k_m.gguf
│   ├── functiongemma-270m-q6_k.gguf
│   └── functiongemma-270m-odg.gguf
└── results/                  ← size / optional lm-eval table

benchmarks/                   ← original dataset files (gitignored)
├── mmlu/{dev,val,test}/*_{split}.csv
├── gsm8k/{train,test}.jsonl
├── hellaswag/hellaswag_val.jsonl
└── arc/challenge/ARC-Challenge-Test.jsonl

models/                       ← aliases for python benchmark.py
evaluation/                   ← HF-free scorer (llama-server, else Ollama)
└── results/comparison.md     ← quality table next to GGUF size
```

---

## How to run

Needs a local GGUF (`ollama pull functiongemma:latest` or a path), `llama-quantize` on `PATH` or `LLAMA_CPP_DIR`, and original test files under `benchmarks/`.

### 1. Produce the GGUFs

```bash
export LLAMA_CPP_DIR=~/.unsloth/llama.cpp   # or your llama.cpp build

odg run --model functiongemma:latest -q q4_k_m --new-run --no-ask
odg experiment prepare --model functiongemma:latest --force
```

`prepare` copies the local source, requantizes uniform Q4 / Q5 / Q6, and builds OpenDynamicGGUF from the latest `recipe.tt`. It does not download from Hugging Face.

### 2. Place or fetch test files

```bash
python benchmark.py fetch
```

Or copy the published files by hand (see [`../evaluation/README.md`](../evaluation/README.md) for URLs). Training splits are used only as few-shot exemplars (MMLU `dev`, GSM8K `train`).

### 3. Score

```bash
# development loop
python benchmark.py \
  --model original.gguf \
  --model q4_k_m.gguf \
  --model opendynamic.gguf

# published comparison
python benchmark.py --suite release \
  --model original.gguf \
  --model q4_k_m.gguf \
  --model opendynamic.gguf
```

Writes `evaluation/results/comparison.md` and `comparison.json`.

The scorer prefers **`llama-server`**. If that binary cannot load the GGUF (for example a vocab-size mismatch against an Ollama blob), it falls back to a matching **Ollama** tag for greedy generation. HellaSwag still needs continuation log-likelihood and is skipped on the Ollama path.

### Optional: lm-eval path

If `lm-eval` is installed, `./benchmark/run_all.sh` still runs `odg experiment prepare|run|compare` and writes `benchmark/results/comparison.md`. Prefer `python benchmark.py` for an HF-free, original-file evaluation.

---

## How to read the table

| Column | Meaning |
|---|---|
| Quality | Task metric on the pinned split (MMLU acc, GSM8K exact match on `####`, HellaSwag acc_norm, ARC-C acc) |
| Compression | GGUF bytes, sitting next to quality so size is never omitted |
| Best quantized | Highest score among Q4 / Q5 / Q6 / OpenDynamicGGUF — **not** vs the source |

Protocols are copied into every result JSON. If two rows were scored with different `--suite` values, the table warns that they are not comparable.

---

## Caveats

1. **Source dtype.** A local Ollama FunctionGemma blob is typically **Q8_0**. Uniform and ODG files are then `--allow-requantize` from that file. That is a valid plumbing comparison; it is **not** a paper-quality BF16 study. Pass `--prefer-hf` on `odg run` when you want original BF16.
2. **No imatrix / held-out KLD** unless `llama-imatrix` and `llama-perplexity` are present. Size and task scores still compare; distributional degradation may not show up on MMLU/GSM8K alone.
3. **llama-server vs Ollama.** This machine’s `llama-server` may reject the Ollama GGUF (`token_embd` 262144 vs expected 262146). Ollama can still load those files. Document which backend produced a table.
4. **FunctionGemma** is not an MMLU/GSM8K model. Low absolute scores are expected; retention vs source is the number that matters.
5. Perplexity on a fixed local corpus and ARC-Easy are not in this suite yet.

---

## Related

- Evaluator and download URLs: [`../evaluation/README.md`](../evaluation/README.md)
- CLI usage: [`../docs/USAGE.md`](../docs/USAGE.md)
- Feature design: [`../docs/platform/02-benchmark-runner.md`](../docs/platform/02-benchmark-runner.md)
- Pipeline: `odg run` → `odg experiment prepare` → `python benchmark.py`
