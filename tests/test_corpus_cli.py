"""Step-07 CLI hard failures for missing template or GGUF."""
from __future__ import annotations

import argparse

import cli
from store import RunStore


def _args(tmp_path, run_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        artifacts=tmp_path,
        model="m",
        run=run_id,
        force=False,
        no_explain=True,
        target_tokens=100,
        seed=42,
        chat_template=None,
        chat_template_file=None,
    )


def _run_with_weight_features(tmp_path, descriptor: dict) -> tuple[RunStore, str]:
    store = RunStore(tmp_path)
    meta = store.create_run("m")
    run_id = meta.run_id
    store.begin_step(run_id, "resolve", {})
    store.complete_step(run_id, "resolve", {"descriptor": descriptor})
    store.begin_step(run_id, "weight_features", {})
    store.complete_step(run_id, "weight_features", {})
    return store, run_id


def test_cmd_corpus_fails_without_template(tmp_path, capsys):
    store, run_id = _run_with_weight_features(tmp_path, {"family": "gemma4"})
    code = cli.cmd_corpus(_args(tmp_path, run_id))
    assert code == 1
    err = capsys.readouterr().err
    assert "needs a real chat template" in err
    assert "--chat-template" in err


def test_cmd_corpus_fails_without_gguf(tmp_path, capsys):
    store, run_id = _run_with_weight_features(
        tmp_path,
        {"family": "gemma4", "chat_template_text": "T", "chat_template_source": "x"},
    )
    code = cli.cmd_corpus(_args(tmp_path, run_id))
    assert code == 1
    err = capsys.readouterr().err
    assert "needs a GGUF" in err
