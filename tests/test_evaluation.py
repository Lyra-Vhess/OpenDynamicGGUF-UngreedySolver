"""HF-free evaluator: original file formats, pinned protocols, no network."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.fetch import fetch_hellaswag, fetch_mmlu, link_models
from evaluation.run import compare, render_markdown
from evaluation.server import letter_logprobs
from evaluation.suites import resolve_limits, suite_limits
from evaluation.tasks import (
    PROTOCOLS,
    extract_choice_letter,
    extract_gsm8k_pred,
    gsm8k_gold,
    iter_arc_challenge,
    iter_gsm8k,
    iter_hellaswag,
    iter_mmlu,
    run_task,
    score_mcq,
)


def _write_mmlu(root: Path) -> None:
    dev = root / "mmlu" / "dev"
    test = root / "mmlu" / "test"
    dev.mkdir(parents=True)
    test.mkdir(parents=True)
    (dev / "anatomy_dev.csv").write_text(
        "Bone?,femur,liver,skin,iris,A\n",
        encoding="utf-8",
    )
    (test / "anatomy_test.csv").write_text(
        "Largest organ?,skin,femur,iris,liver,A\n"
        "Pump?,heart,skin,bone,hair,A\n",
        encoding="utf-8",
    )


def _write_gsm8k(root: Path) -> None:
    d = root / "gsm8k"
    d.mkdir(parents=True)
    train = [
        {
            "question": "What is 1+1?",
            "answer": "1+1=2\n#### 2",
        }
    ]
    test = [
        {"question": "What is 6*7?", "answer": "6*7=42\n#### 42"},
        {"question": "What is 3+4?", "answer": "#### 7"},
    ]
    (d / "train.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in train), encoding="utf-8"
    )
    (d / "test.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in test), encoding="utf-8"
    )


def _write_hellaswag(root: Path) -> None:
    d = root / "hellaswag"
    d.mkdir(parents=True)
    rows = [
        {
            "ind": 1,
            "ctx": "A person opens the fridge and",
            "endings": ["takes out milk.", "launches a rocket.", "solves a PDE.", "melts."],
            "label": 0,
        },
        {
            "ind": 2,
            "ctx": "The chef chops onions and",
            "endings": ["flies away.", "cries from the fumes.", "builds a dam.", "hibernates."],
            "label": 1,
        },
        {"ind": 3, "ctx": "unlabeled", "endings": ["a", "b", "c", "d"]},  # no label → skip
    ]
    (d / "hellaswag_val.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8"
    )


def _write_arc(root: Path) -> None:
    d = root / "arc" / "challenge"
    d.mkdir(parents=True)
    ex = {
        "id": "ARC-1",
        "question": {
            "stem": "Which is a solid?",
            "choices": [
                {"label": "A", "text": "ice"},
                {"label": "B", "text": "steam"},
                {"label": "C", "text": "air"},
                {"label": "D", "text": "fog"},
            ],
        },
        "answerKey": "A",
    }
    (d / "ARC-Challenge-Test.jsonl").write_text(json.dumps(ex) + "\n", encoding="utf-8")


class FakeServer:
    supports_loglikelihood = True
    def next_logprobs(self, prompt, top=128):
        return {" A": -0.1, "A": -0.2, " B": -2.0, " C": -3.0, " D": -4.0, "id:1": -0.1}

    def generate(self, prompt, n_predict=256, stop=None):
        if "6*7" in str(prompt):
            return "Let's think. 6*7=42\n#### 42"
        return "#### 0"

    def loglikelihood(self, context, continuation):
        text = (context + continuation).lower()
        if continuation.strip() in {"A", "B", "C", "D"}:
            # Prefer A
            scores = {"A": -0.1, "B": -2.0, "C": -3.0, "D": -4.0}
            return scores[continuation.strip()], 1
        if "milk" in continuation.lower() or "cries" in continuation.lower():
            return -0.2, max(len(continuation.split()), 1)
        return -5.0, max(len(continuation.split()), 1)


def test_mmlu_reads_hendrycks_csv_and_keeps_abcd(tmp_path):
    _write_mmlu(tmp_path)
    items = list(iter_mmlu(tmp_path))
    assert len(items) == 2
    assert items[0]["gold"] == "A"
    assert "A. skin" in items[0]["prompt"]
    assert items[0]["prompt"].endswith("Answer:")
    assert "multiple choice questions" in items[0]["prompt"]
    # 5-shot block from original-format dev CSV is inlined, not rewritten
    assert "Bone?" in items[0]["prompt"]


def test_gsm8k_keeps_hash_gold_and_parses_pred(tmp_path):
    _write_gsm8k(tmp_path)
    items = list(iter_gsm8k(tmp_path))
    assert items[0]["gold"] == "42"
    assert items[0]["prompt"].startswith("Question: What is 1+1?")
    assert gsm8k_gold("foo #### 1,234") == "1234"
    assert extract_gsm8k_pred("reason\n#### 42\n") == "42"
    assert extract_gsm8k_pred("the answer is 7") == "7"


def test_hellaswag_scores_val_and_skips_unlabeled(tmp_path):
    _write_hellaswag(tmp_path)
    items = list(iter_hellaswag(tmp_path))
    assert len(items) == 2
    assert items[0]["gold"] == 0
    assert items[0]["context"].endswith(" ")
    assert len(items[0]["endings"]) == 4


def test_arc_keeps_ai2_answer_key(tmp_path):
    _write_arc(tmp_path)
    items = list(iter_arc_challenge(tmp_path))
    assert items[0]["gold"] == "A"
    assert items[0]["labels"] == ["A", "B", "C", "D"]
    assert "A. ice" in items[0]["prompt"]


def test_dev_suite_is_the_fast_loop_caps():
    caps = suite_limits("dev")
    assert caps == {
        "mmlu": 500,
        "gsm8k": 200,
        "hellaswag": 500,
        "arc_challenge": 200,
    }
    assert resolve_limits("release") == {
        "mmlu": None,
        "gsm8k": None,
        "hellaswag": None,
        "arc_challenge": None,
    }
    assert resolve_limits("dev", limit=32)["mmlu"] == 32
    assert resolve_limits("dev", limit=0)["gsm8k"] is None


def test_iterators_stop_at_first_n_of_the_official_split(tmp_path):
    _write_mmlu(tmp_path)
    assert [x["gold"] for x in iter_mmlu(tmp_path, limit=1)] == ["A"]
    assert len(list(iter_mmlu(tmp_path, limit=500))) == 2  # fixture only has 2


def test_protocols_are_pinned():
    assert PROTOCOLS["mmlu"]["shots"] == 5
    assert PROTOCOLS["gsm8k"]["shots"] == 8
    assert PROTOCOLS["hellaswag"]["split"] == "val"
    assert PROTOCOLS["hellaswag"]["metric"] == "acc_norm"
    assert PROTOCOLS["arc_challenge"]["split"] == "test"


def test_letter_logprobs_collapses_tokenizer_variants():
    dist = {" A": -0.2, "A": -0.4, "B.": -1.0, " C\n": -3.0}
    scored = letter_logprobs(dist)
    assert scored["A"] == -0.2
    assert scored["B"] == -1.0
    assert scored["C"] == -3.0
    assert scored["D"] == float("-inf")
    assert extract_choice_letter("B.", ["A", "B", "C", "D"]) == "B"
    assert extract_choice_letter("The answer is C\n", ["A", "B", "C", "D"]) == "C"


def test_run_task_against_original_fixtures(tmp_path):
    _write_mmlu(tmp_path)
    _write_gsm8k(tmp_path)
    _write_hellaswag(tmp_path)
    _write_arc(tmp_path)
    server = FakeServer()
    mmlu = run_task(server, "mmlu", limit=2, root=tmp_path)
    assert mmlu.n == 2 and mmlu.score == 1.0
    gsm = run_task(server, "gsm8k", limit=1, root=tmp_path)
    assert gsm.n == 1 and gsm.score == 1.0
    hs = run_task(server, "hellaswag", limit=2, root=tmp_path)
    assert hs.n == 2 and hs.metric == "acc_norm" and hs.score == 1.0
    arc = run_task(server, "arc_challenge", limit=1, root=tmp_path)
    assert arc.n == 1 and arc.score == 1.0
    assert score_mcq(server, "Answer:", "A", ("A", "B", "C", "D")) is True


def test_compare_table_marks_best_quantized():
    results = [
        {
            "label": "Source",
            "bytes": 287_000_000,
            "gguf": "original.gguf",
            "tasks": {"mmlu": {"score": 0.40}, "gsm8k": {"score": 0.10},
                      "hellaswag": {"score": 0.50}, "arc_challenge": {"score": 0.30}},
        },
        {
            "label": "Q4_K_M",
            "bytes": 250_000_000,
            "gguf": "q4_k_m.gguf",
            "tasks": {"mmlu": {"score": 0.30}, "gsm8k": {"score": 0.05},
                      "hellaswag": {"score": 0.40}, "arc_challenge": {"score": 0.20}},
        },
        {
            "label": "OpenDynamicGGUF",
            "bytes": 245_000_000,
            "gguf": "opendynamic.gguf",
            "tasks": {"mmlu": {"score": 0.35}, "gsm8k": {"score": 0.08},
                      "hellaswag": {"score": 0.45}, "arc_challenge": {"score": 0.25}},
        },
    ]
    comp = compare(results)
    mmlu = next(r for r in comp["quality"] if r["task"] == "mmlu")
    assert mmlu["best_quantized"] == "OpenDynamicGGUF"
    md = render_markdown(comp)
    assert "MMLU" in md and "**35.0**" in md
    assert "original datasets" in md.lower()
    assert "5-shot" in md
    mixed = compare(
        [
            {**results[0], "suite": "dev"},
            {**results[2], "suite": "release"},
        ]
    )
    assert mixed["notes"] and "Mixed suites" in mixed["notes"][0]
    assert "not comparable" in render_markdown(mixed)


def test_fetch_skips_when_original_files_exist(tmp_path):
    _write_mmlu(tmp_path)
    out = fetch_mmlu(tmp_path)
    assert out == tmp_path / "mmlu"
    # still the Hendrycks-format CSV we wrote, not a converted JSON
    assert (tmp_path / "mmlu" / "test" / "anatomy_test.csv").is_file()
    _write_hellaswag(tmp_path)
    fetch_hellaswag(tmp_path)
    assert (tmp_path / "hellaswag" / "hellaswag_val.jsonl").is_file()


def test_link_models_aliases_prepared_ggufs(tmp_path):
    src = tmp_path / "benchmark" / "models"
    src.mkdir(parents=True)
    (src / "functiongemma-270m-bf16.gguf").write_bytes(b"GGUF")
    (src / "functiongemma-270m-odg.gguf").write_bytes(b"GGUF")
    link_models(tmp_path)
    assert (tmp_path / "models" / "original.gguf").is_file()
    assert (tmp_path / "models" / "opendynamic.gguf").is_file()
    assert not (tmp_path / "models" / "q4_k_m.gguf").exists()
