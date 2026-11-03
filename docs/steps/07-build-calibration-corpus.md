# Step 07 — Build the calibration corpus (3-way split)

← [06 Weight features](./06-compute-weight-features.md) · [Index](./README.md) · Next: [08 Activation features](./08-compute-activation-features.md) →

---

## Goal

Create the **text** the model will be run on for activation stats, imatrix, KL search, and final validation — and split it so search never sees the judging set.

---

## Command

```bash
odg corpus --model functiongemma:latest
# larger production-sized corpus:
odg corpus --model functiongemma:latest --target-tokens 300000 --force
odg status --model functiongemma:latest

# explicit template override:
odg corpus --model my-model --chat-template-file ./chat_template.jinja
odg corpus --model my-model --chat-template chatml
```

Requires Step 06 done and a GGUF for the model (inside `odg run`, that is
the step-09 frozen GGUF). It also needs `llama-server` from llama.cpp.
Default `--target-tokens 50000` (plumbing). Use `300000+` for production.

---

## Why it exists

Activation stats, imatrix, and ΔKLD all need **real prompts**. Wikipedia-only calibration overfits Wikipedia-style KL scores. Instruct models also need **chat-template** formatting.

---

## Inputs / Outputs

| | |
|---|---|
| **Input** | Resolve descriptor (`chat_template_text`, `chat_template_source`, `chat_template_sha256`, `specialty_domain`) and a GGUF |
| **Output** | `calib.txt`, `search.txt`, `heldout.txt`, `corpus_manifest.json`, `chat_template.jinja` |

```text
steps/07_corpus/
  calib.txt             # ~60% — imatrix + activation features
  search.txt            # ~20% — ΔKLD during probe / optimize
  heldout.txt           # ~20% — validation ONLY
  corpus_manifest.json
  chat_template.jinja   # exact template used for this corpus
  llama-server.log      # renderer log
  output.json
  status.json
  log.txt
```

Hard rule: **optimizer must not read held-out.**

---

## How it works

Mixed offline banks (no download): conversation / code / math / multilingual / specialty.

Every document is an OpenAI-style message list rendered through the model's
own template. Step 07 starts `llama-server` with the GGUF and the resolved
template file, then calls `/apply-template` for each document with
`add_generation_prompt=false`. The server runs with
`--no-prefill-assistant`, so a final assistant message is closed rather than
left open for generation. Its output omits the leading BOS when the
tokenizer adds BOS, so downstream tools add exactly one.

There is no plain-text fallback and no family-name guess. A model with no
template stops the step. Use `--chat-template` or `--chat-template-file` to
supply one explicitly.

For FunctionGemma (`specialty_domain=function_calling`), the corpus includes
tool-call traces, which the model's template renders in its own format.

Then tiles with light variants until `--target-tokens`, and splits 60/20/20
(`seed=42`).

---

## Done when

- [x] Three files written with disjoint content
- [x] Every document rendered through the model's live chat template
- [x] Template source and SHA-256 recorded in the corpus output and manifest
- [x] Domain data included when specialty_domain is set
- [x] Token counts logged in a corpus manifest

## Next

[Step 08 — Compute activation features](./08-compute-activation-features.md)
