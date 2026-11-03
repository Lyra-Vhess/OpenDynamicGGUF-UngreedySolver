"""Step 07 renders every document through the live template."""
from __future__ import annotations

from pathlib import Path

import pytest

from chat_template import TemplateInfo
from corpus import build_corpus


def test_build_corpus_requires_usable_template(tmp_path):
    def render(doc):
        return "unused"

    with pytest.raises(ValueError, match="without a usable chat template"):
        build_corpus(
            model_ref="m",
            out_dir=tmp_path,
            template=TemplateInfo(),
            render_document=render,
            target_tokens=100,
        )


def test_build_corpus_uses_renderer_and_records_provenance(tmp_path):
    seen: list[dict] = []

    def render(doc):
        seen.append(doc)
        user = str(doc.get("user") or "")
        assistant = str(doc.get("assistant") or "")
        return f"<turn>user\n{user}\n<turn>model\n{assistant}\n"

    template = TemplateInfo(
        label="gemma4",
        text="TEMPLATE",
        source="local:chat_template.jinja",
        sha256="abc123",
        variant=None,
        add_bos_token=True,
    )
    result, manifest = build_corpus(
        model_ref="m",
        out_dir=tmp_path,
        template=template,
        render_document=render,
        target_tokens=500,
        seed=42,
    )
    assert seen
    assert result.n_documents > 0
    assert result.chat_template == "gemma4"
    assert result.chat_template_source == "local:chat_template.jinja"
    assert result.chat_template_sha256 == "abc123"
    assert result.add_bos_token is True
    held = (tmp_path / "heldout.txt").read_text(encoding="utf-8")
    assert "<turn>user" in held
    assert "User:" not in held
    assert manifest["chat_template_source"] == "local:chat_template.jinja"
    assert manifest["chat_template_sha256"] == "abc123"
