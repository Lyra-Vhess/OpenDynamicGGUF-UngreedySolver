"""Step 01 chat-template resolution and overrides."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import resolve
from resolve import (
    ArchitectureDescriptor,
    _merge_desc,
    _resolve_local_gguf,
    inspect_local_dir,
    resolve_model,
)


def _local_model(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "num_hidden_layers": 42,
            "hidden_size": 2560,
        }),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"")
    return root


def test_inspect_local_dir_loads_jinja_template(tmp_path):
    root = _local_model(tmp_path)
    (root / "chat_template.jinja").write_text("JINJA", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({
            "chat_template": "INLINE",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
        }),
        encoding="utf-8",
    )
    desc, full_prec = inspect_local_dir(root)
    assert full_prec is True
    assert desc.family == "gemma4"
    assert desc.chat_template == "gemma4"
    assert desc.chat_template_text == "JINJA"
    assert desc.chat_template_source == "local:chat_template.jinja"
    assert desc.chat_template_sha256
    assert desc.bos_token == "<bos>"
    assert desc.eos_token == "<eos>"


def test_inspect_local_dir_no_template_is_not_a_family_label(tmp_path):
    root = _local_model(tmp_path)
    desc, _ = inspect_local_dir(root)
    assert desc.chat_template is None
    assert desc.chat_template_text is None


def test_local_gguf_reads_embedded_template(tmp_path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    info = {
        "architecture": "gemma4",
        "layer_count": 42,
        "embedding_length": 2560,
        "parameter_count": 1000,
        "dtype_summary": {"BF16": 1},
        "metadata": {
            "tokenizer.chat_template": "GGUF TEMPLATE",
            "tokenizer.ggml.add_bos_token": True,
        },
    }
    monkeypatch.setattr("load.open_gguf", lambda path: info)
    result = _resolve_local_gguf("m.gguf", gguf, [])
    desc = result.descriptor
    assert desc.chat_template_text == "GGUF TEMPLATE"
    assert desc.chat_template_source == "gguf:tokenizer.chat_template"
    assert desc.add_bos_token is True
    assert desc.chat_template_sha256


def test_merge_keeps_base_template_when_hf_has_none():
    base = ArchitectureDescriptor(
        family="gemma4",
        chat_template="gemma4",
        chat_template_text="BASE",
        chat_template_source="gguf:tokenizer.chat_template",
        chat_template_sha256="abc",
    )
    hf = ArchitectureDescriptor(family="gemma4")
    merged = _merge_desc(base, hf)
    assert merged.chat_template_text == "BASE"
    assert merged.chat_template_source == "gguf:tokenizer.chat_template"


def test_resolve_model_cli_override(tmp_path):
    root = _local_model(tmp_path)
    result = resolve_model(
        str(root),
        chat_template="{% for m in messages %}{{ m.content }}{% endfor %}",
    )
    desc = result.descriptor
    assert desc.chat_template_source == "cli:inline"
    assert desc.chat_template_text
    assert desc.chat_template_sha256


def test_resolve_model_builtin_override(tmp_path):
    root = _local_model(tmp_path)
    result = resolve_model(str(root), chat_template="chatml")
    desc = result.descriptor
    assert desc.chat_template_builtin == "chatml"
    assert desc.chat_template_source == "cli:builtin:chatml"
    assert desc.chat_template_sha256
