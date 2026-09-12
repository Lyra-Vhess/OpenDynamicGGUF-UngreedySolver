# Local evaluation (no Hugging Face)

Test splits only. Training data is never scored. Drop the original files into
`benchmarks/` later — this evaluator reads them as published (CSV / JSONL),
not a converted schema.

## Two suites

| Suite | MMLU | GSM8K | HellaSwag | ARC-Challenge | When |
|---|---:|---:|---:|---:|---|
| **`dev`** (default) | 500 | 200 | 500 | 200 | every quantizer change |
| **`release`** | full ~14K | full 1,319 | full ~10K val | full 1,172 | published comparison |

Subsets are the **first N rows** of the official file order, so every GGUF
sees the same questions. ~1.4K items in `dev`, ~26K in `release`.

Perplexity on a fixed local corpus is not in this suite yet.

```bash
# After original test files are in benchmarks/ and GGUFs are in models/:
python benchmark.py --model original.gguf --model opendynamic.gguf
python benchmark.py --suite release --model original.gguf --model opendynamic.gguf
```

Writes `evaluation/results/comparison.md`. Do not compare a `dev` table to a
`release` table.

## Layout (files you place later)

```text
benchmarks/
├── mmlu/test/                 # Hendrycks *_test.csv  (+ mmlu/dev/ for 5-shot)
├── gsm8k/test.jsonl           # OpenAI 1,319 test rows (train.jsonl only for shots)
├── hellaswag/hellaswag_val.jsonl
└── arc/challenge/ARC-Challenge-Test.jsonl
models/
├── original.gguf
├── q4_k_m.gguf
└── opendynamic.gguf
```

Optional: `python benchmark.py fetch` downloads those originals. Not required
if you copy the files in by hand.

`llama-server` must be on `PATH` or under `LLAMA_CPP_DIR`.
