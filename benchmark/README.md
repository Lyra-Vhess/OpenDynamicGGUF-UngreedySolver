# Same-model experiment (FunctionGemma 270M)

Everything except quantization is identical.

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
        ARC         TruthfulQA  ...
```

Default model: [`google/functiongemma-270m-it`](https://huggingface.co/google/functiongemma-270m-it).

Variants: **BF16** · **Q4_K_M** · **Q5_K_M** · **Q6_K** · **OpenDynamicGGUF**.

Pinned (see `config.json`): task list, harness, seed `0`, batch size `8`, no global `--num_fewshot` (each task keeps the EleutherAI default). Tokenizer for GGUF eval is the original HF tokenizer.

## One command

```bash
# gated Gemma weights:  huggingface-cli login   (or HF_TOKEN)
# llama.cpp:            export LLAMA_CPP_DIR=~/llama.cpp
# harness:              pip install 'lm-eval[hf]' transformers torch llama-cpp-python

./benchmark/run_all.sh
```

That prepares the GGUFs, runs every variant, and writes:

```text
benchmark/results/comparison.md
benchmark/results/comparison.json
```

## Step by step

```bash
./benchmark/prepare.sh          # HF snapshot + BF16 GGUF + uniform quants + ODG
./benchmark/run_bf16.sh         # original BF16 via --model hf
./benchmark/run_q4.sh
./benchmark/run_q5.sh
./benchmark/run_q6.sh
./benchmark/run_odg.sh
python benchmark/compare.py     # one table
```

`dev` suite (default) caps each task at 32 samples so a 270M model finishes on a laptop. Full tasks:

```bash
./benchmark/run_all.sh --suite paper
```

Same flags work on `odg experiment prepare|run|compare`.

## What the table must include

Quality (MMLU, GSM8K, HellaSwag, ARC-C, TruthfulQA), compression (GGUF size, ratio vs BF16, bytes/parameter), behavior (perplexity, KL vs BF16, token agreement), inference (prompt tok/s, gen tok/s).

A better score at a much larger file than Q4_K_M is not an impressive quantization result. The interesting claim is quality **at similar size**.

FunctionGemma 270M is a function-calling specialist — absolute MMLU/GSM8K will be low. Compare **retention vs BF16**, not SOTA.

Do not treat a single GGUF lm-eval number as authoritative ([harness issue 2887](https://github.com/EleutherAI/lm-evaluation-harness/issues/2887)).
