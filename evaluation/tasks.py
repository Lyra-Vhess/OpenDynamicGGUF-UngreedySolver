"""Load original benchmark files and score them. Files are not rewritten."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from evaluation.server import LlamaServer

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"

# Pinned protocols (recorded in every result).
PROTOCOLS: dict[str, dict[str, Any]] = {
    "mmlu": {
        "split": "test",
        "shots": 5,
        "metric": "acc",
        "scoring": "next-token logprob over A/B/C/D",
        "source": "https://github.com/hendrycks/test + people.eecs.berkeley.edu/~hendrycks/data.tar",
    },
    "gsm8k": {
        "split": "test",
        "shots": 8,
        "metric": "exact_match",
        "scoring": "greedy generate, parse #### number (OpenAI GSM8K); 8-shot = first 8 of official train.jsonl",
        "source": "https://github.com/openai/grade-school-math",
    },
    "hellaswag": {
        "split": "val",
        "shots": 0,
        "metric": "acc_norm",
        "scoring": "0-shot ending loglikelihood, length-normalized (original HellaSwag)",
        "source": "https://github.com/rowanz/hellaswag  (test labels are hidden; val is the labeled split)",
    },
    "arc_challenge": {
        "split": "test",
        "shots": 0,
        "metric": "acc",
        "scoring": "next-token logprob over answerKey letters (AI2 ARC-Challenge)",
        "source": "https://allenai.org/data/arc  ARC-V1-Feb2018",
    },
}


def format_subject(subject: str) -> str:
    return " ".join(subject.replace("_", " ").split())


def mmlu_example_prompt(question: str, a: str, b: str, c: str, d: str, answer: str | None = None) -> str:
    block = f"{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"
    if answer is not None:
        block += f" {answer.strip()}\n\n"
    return block


def load_mmlu_csv(path: Path) -> list[tuple[str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 6:
                continue
            rows.append((row[0], row[1], row[2], row[3], row[4], row[5].strip()))
    return rows


def iter_mmlu(root: Path = BENCH, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    test_dir = root / "mmlu" / "test"
    dev_dir = root / "mmlu" / "dev"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"MMLU test CSVs missing: {test_dir} (run: python evaluation/run.py fetch)")
    n = 0
    for csv_path in sorted(test_dir.glob("*_test.csv")):
        subject = csv_path.name[: -len("_test.csv")]
        dev_path = dev_dir / f"{subject}_dev.csv"
        shots = load_mmlu_csv(dev_path)[: PROTOCOLS["mmlu"]["shots"]] if dev_path.is_file() else []
        header = (
            "The following are multiple choice questions (with answers) about "
            f"{format_subject(subject)}.\n\n"
        )
        shot_text = "".join(mmlu_example_prompt(*s[:5], s[5]) for s in shots)
        for q, a, b, c, d, ans in load_mmlu_csv(csv_path):
            prompt = header + shot_text + mmlu_example_prompt(q, a, b, c, d)
            yield {
                "id": f"{subject}:{n}",
                "subject": subject,
                "prompt": prompt,
                "gold": ans.upper()[:1],
            }
            n += 1
            if limit is not None and n >= limit:
                return


def gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return answer.strip().replace(",", "")


def extract_gsm8k_pred(text: str) -> str:
    hashes = re.findall(r"####\s*(-?[\d,]+)", text)
    if hashes:
        return hashes[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else ""


def iter_gsm8k(root: Path = BENCH, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    test = root / "gsm8k" / "test.jsonl"
    train = root / "gsm8k" / "train.jsonl"
    if not test.is_file():
        raise FileNotFoundError(f"GSM8K test JSONL missing: {test}")
    shots: list[dict[str, str]] = []
    if train.is_file():
        with train.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= PROTOCOLS["gsm8k"]["shots"]:
                    break
                shots.append(json.loads(line))
    shot_text = "".join(
        f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots
    )
    with test.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            ex = json.loads(line)
            yield {
                "id": f"gsm8k:{i}",
                "prompt": shot_text + f"Question: {ex['question']}\nAnswer:",
                "gold": gsm8k_gold(ex["answer"]),
            }
            if limit is not None and i + 1 >= limit:
                return


def iter_hellaswag(root: Path = BENCH, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    # Original test.jsonl has no public labels. Score the original val split.
    path = root / "hellaswag" / "hellaswag_val.jsonl"
    if not path.is_file():
        path = root / "hellaswag" / "val.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"HellaSwag val JSONL missing under {root / 'hellaswag'}")
    with path.open(encoding="utf-8") as fh:
        n = 0
        for line in fh:
            ex = json.loads(line)
            if "label" not in ex:
                continue
            ctx = (ex.get("ctx") or (str(ex.get("ctx_a") or "") + " " + str(ex.get("ctx_b") or ""))).rstrip()
            yield {
                "id": f"hellaswag:{ex.get('ind', n)}",
                "context": ctx + " ",
                "endings": list(ex["endings"]),
                "gold": int(ex["label"]),
            }
            n += 1
            if limit is not None and n >= limit:
                return


def iter_arc_challenge(root: Path = BENCH, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    path = root / "arc" / "challenge" / "ARC-Challenge-Test.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"ARC-Challenge test JSONL missing: {path}")
    with path.open(encoding="utf-8") as fh:
        n = 0
        for line in fh:
            ex = json.loads(line)
            stem = ex["question"]["stem"]
            choices = ex["question"]["choices"]
            labels = [c["label"] for c in choices]
            texts = [c["text"] for c in choices]
            lines = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
            prompt = f"Question: {stem}\n{lines}\nAnswer:"
            yield {
                "id": ex.get("id") or f"arc:{n}",
                "prompt": prompt,
                "gold": str(ex["answerKey"]).strip().upper(),
                "labels": labels,
            }
            n += 1
            if limit is not None and n >= limit:
                return


def extract_choice_letter(text: str, labels: tuple[str, ...] | list[str]) -> str:
    labs = [str(x).strip().upper() for x in labels]
    blob = (text or "").strip().upper()
    for lab in labs:
        if blob.startswith(lab) and (len(blob) == len(lab) or not blob[len(lab)].isalnum()):
            return lab
    for lab in labs:
        m = re.search(rf"\b{re.escape(lab)}\b", blob)
        if m:
            return lab
    return ""


def score_mcq(server: LlamaServer, prompt: str, gold: str, labels: tuple[str, ...] | list[str]) -> bool:
    gold_l = str(gold).strip().upper()
    if getattr(server, "supports_loglikelihood", True):
        scored: dict[str, float] = {}
        for lab in labels:
            letter = str(lab).strip().upper()
            ll, _n = server.loglikelihood(prompt, f" {letter}")
            scored[letter] = ll
        pred = max(scored, key=scored.get)
        return pred == gold_l
    # Hendrycks-style generate the answer letter (Ollama has no logprobs).
    text = server.generate(prompt, n_predict=8, stop=["\n"])
    return extract_choice_letter(text, labels) == gold_l


def score_hellaswag(server: LlamaServer, context: str, endings: list[str], gold: int) -> tuple[bool, bool]:
    raw: list[float] = []
    norm: list[float] = []
    for ending in endings:
        ll, ntok = server.loglikelihood(context, ending)
        raw.append(ll)
        norm.append(ll / max(ntok, 1))
    acc = int(raw.index(max(raw)) == gold)
    acc_norm = int(norm.index(max(norm)) == gold)
    return bool(acc), bool(acc_norm)


@dataclass
class TaskResult:
    name: str
    metric: str
    score: float
    n: int
    protocol: dict[str, Any]
    extra: dict[str, Any]


def run_task(
    server: LlamaServer,
    name: str,
    *,
    limit: int | None,
    root: Path = BENCH,
) -> TaskResult:
    proto = PROTOCOLS[name]
    n_ok = 0
    n = 0
    extra: dict[str, Any] = {}
    if name == "mmlu":
        for ex in iter_mmlu(root, limit=limit):
            n_ok += int(score_mcq(server, ex["prompt"], ex["gold"], ("A", "B", "C", "D")))
            n += 1
    elif name == "gsm8k":
        for ex in iter_gsm8k(root, limit=limit):
            text = server.generate(ex["prompt"], n_predict=256, stop=["\nQuestion:"])
            n_ok += int(extract_gsm8k_pred(text) == ex["gold"])
            n += 1
    elif name == "hellaswag":
        if not getattr(server, "supports_loglikelihood", True):
            raise RuntimeError(
                "HellaSwag needs ending loglikelihood; this backend only supports greedy generate"
            )
        acc = acc_n = 0
        for ex in iter_hellaswag(root, limit=limit):
            a, an = score_hellaswag(server, ex["context"], ex["endings"], ex["gold"])
            acc += int(a)
            acc_n += int(an)
            n += 1
        extra["acc"] = acc / n if n else 0.0
        n_ok = acc_n
    elif name == "arc_challenge":
        for ex in iter_arc_challenge(root, limit=limit):
            n_ok += int(score_mcq(server, ex["prompt"], ex["gold"], ex["labels"]))
            n += 1
    else:
        raise ValueError(f"Unknown task {name}")
    score = n_ok / n if n else 0.0
    return TaskResult(name=name, metric=proto["metric"], score=score, n=n, protocol=proto, extra=extra)
