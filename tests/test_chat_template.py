"""Chat-template discovery, overrides, and llama.cpp request shape."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import chat_template
from chat_template import (
    LlamaTemplateRenderer,
    TemplateInfo,
    apply_override,
    document_messages,
    load_local_template,
    select_variant,
)


def test_select_variant_string_list_and_default():
    assert select_variant("hello") == ("hello", None)
    assert select_variant([
        {"name": "tool", "template": "tool"},
        {"name": "default", "template": "def"},
        {"name": "other", "template": "other"},
    ]) == ("def", "default")
    assert select_variant([{"name": "first", "template": "one"}]) == ("one", "first")
    assert select_variant({"default": "d", "other": "o"}) == ("d", "default")
    assert select_variant(None) == (None, None)


def test_local_template_jinja_has_priority(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("JINJA {{ x }}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({
            "chat_template": "INLINE {{ x }}",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "add_bos_token": True,
        }),
        encoding="utf-8",
    )
    info = load_local_template(tmp_path, label="gemma4")
    assert info is not None
    assert info.text == "JINJA {{ x }}"
    assert info.source == "local:chat_template.jinja"
    assert info.sha256
    assert info.bos_token == "<bos>"
    assert info.eos_token == "<eos>"
    assert info.add_bos_token is True


def test_local_template_inline_variant(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({
            "chat_template": [
                {"name": "default", "template": "DEFAULT"},
                {"name": "tool", "template": "TOOL"},
            ]
        }),
        encoding="utf-8",
    )
    info = load_local_template(tmp_path, label="m")
    assert info is not None
    assert info.text == "DEFAULT"
    assert info.variant == "default"
    assert info.source == "local:tokenizer_config.json"


def test_local_template_chat_template_json(tmp_path):
    (tmp_path / "chat_template.json").write_text(
        json.dumps({"chat_template": "FROM JSON"}), encoding="utf-8"
    )
    info = load_local_template(tmp_path, label="m")
    assert info is not None
    assert info.text == "FROM JSON"
    assert info.source == "local:chat_template.json"


def test_no_template_returns_none(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert load_local_template(tmp_path, label="m") is None


def test_apply_override_inline_file_and_builtin(tmp_path):
    inline = apply_override("{% for m in messages %}{{ m.content }}{% endfor %}")
    assert inline is not None
    assert inline.source == "cli:inline"
    assert inline.text and "messages" in inline.text

    path = tmp_path / "t.jinja"
    path.write_text("FILE", encoding="utf-8")
    file_info = apply_override(None, path)
    assert file_info is not None
    assert file_info.source == "cli:t.jinja"
    assert file_info.text == "FILE"

    builtin = apply_override("chatml", label="m")
    assert builtin is not None
    assert builtin.source == "cli:builtin:chatml"
    assert builtin.builtin == "chatml"
    assert builtin.text is None
    assert builtin.usable

    with pytest.raises(ValueError, match="not both"):
        apply_override("chatml", path)
    with pytest.raises(FileNotFoundError):
        apply_override(None, tmp_path / "missing.jinja")


def test_document_messages_shapes():
    assert document_messages({"user": "hi"}) == [
        {"role": "user", "content": "hi"}
    ]
    assert document_messages({"user": "hi", "assistant": "yo"}) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert document_messages({
        "turns": [
            {"role": "system", "content": "sys"},
            {"role": "model", "content": "answer"},
        ]
    }) == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "answer"},
    ]
    with pytest.raises(ValueError, match="no non-empty"):
        document_messages({"user": "  "})


def test_renderer_command_includes_llama_cpp_contract(monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    tpl = tmp_path / "t.jinja"
    tpl.write_text("x", encoding="utf-8")
    info = TemplateInfo(text="x", source="local:t.jinja", sha256="abc")
    renderer = LlamaTemplateRenderer(
        model_gguf=model,
        template=info,
        template_path=tpl,
        server_binary="/usr/bin/llama-server",
    )
    renderer.port = 12345
    cmd = renderer._command()
    assert cmd[:3] == ["/usr/bin/llama-server", "-m", str(model)]
    assert "--no-prefill-assistant" in cmd
    assert "--jinja" in cmd
    assert "--chat-template-file" in cmd
    assert str(tpl) in cmd


def test_renderer_command_builtin(monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    info = TemplateInfo(source="cli:builtin:chatml", builtin="chatml")
    renderer = LlamaTemplateRenderer(
        model_gguf=model,
        template=info,
        server_binary="/usr/bin/llama-server",
    )
    renderer.port = 12345
    cmd = renderer._command()
    assert "--chat-template" in cmd
    assert "chatml" in cmd
    assert "--chat-template-file" not in cmd


def test_renderer_requires_template(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    renderer = LlamaTemplateRenderer(
        model_gguf=model,
        template=TemplateInfo(),
        server_binary="/usr/bin/llama-server",
    )
    with pytest.raises(RuntimeError, match="No chat template"):
        renderer.__enter__()


def test_render_messages_posts_and_reads_prompt(monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    info = TemplateInfo(text="x", source="local:t.jinja", sha256="abc")
    renderer = LlamaTemplateRenderer(model_gguf=model, template=info)
    renderer._url = "http://127.0.0.1:1/apply-template"
    seen = {}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'{"prompt": "rendered"}'

    def fake_urlopen(request, timeout=30):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(chat_template.urllib.request, "urlopen", fake_urlopen)
    prompt = renderer.render_messages([{"role": "user", "content": "hi"}])
    assert prompt == "rendered"
    assert seen["body"]["add_generation_prompt"] is False
    assert seen["body"]["messages"][0]["content"] == "hi"
