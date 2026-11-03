"""Chat-template discovery and llama.cpp rendering.

The model's own template text is the source of truth.  Step 01 resolves
it from local files, GGUF metadata, or an explicit override.  Step 07
renders corpus documents with llama.cpp's Jinja engine through
``llama-server /apply-template``, so the text that reaches imatrix and
perplexity is exactly what llama.cpp would send to the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request

from llama_bins import find_llama_binary


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        return {"chat_template": data}
    return {}


def _token_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
        token = value.get("token")
        if isinstance(token, str):
            return token
    return None


def select_variant(value: Any) -> tuple[str | None, str | None]:
    """Return ``(template_text, variant_name)`` from a HF template value.

    Hugging Face accepts a string, a list of named templates, or a mapping.
    Prefer the entry named ``default``; otherwise use the first.
    """
    if isinstance(value, str):
        text = value.strip()
        return (text, None) if text else (None, None)

    if isinstance(value, dict):
        if isinstance(value.get("template"), str):
            name = value.get("name")
            text = value["template"].strip()
            return (text, str(name) if name else None)
        for name in ("default", "chat_template"):
            if name in value:
                text, variant = select_variant(value[name])
                return text, variant or (name if name != "default" else "default")
        for name, item in value.items():
            text, variant = select_variant(item)
            if text:
                return text, variant or str(name)
        return None, None

    if isinstance(value, (list, tuple)):
        entries = [
            item for item in value
            if isinstance(item, dict) and isinstance(item.get("template"), str)
        ]
        for item in entries:
            if str(item.get("name", "")).lower() == "default":
                return item["template"].strip(), str(item.get("name") or "default")
        if entries:
            item = entries[0]
            return item["template"].strip(), str(item.get("name") or "0")
        if value and isinstance(value[0], str):
            text = value[0].strip()
            return (text, None) if text else (None, None)
    return None, None


@dataclass
class TemplateInfo:
    """Template text plus enough provenance to reproduce a corpus."""

    label: str | None = None
    text: str | None = None
    source: str = "none"
    sha256: str | None = None
    variant: str | None = None
    builtin: str | None = None
    bos_token: str | None = None
    eos_token: str | None = None
    add_bos_token: bool | None = None

    @property
    def usable(self) -> bool:
        return bool(self.text or self.builtin)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_descriptor(cls, desc: dict[str, Any] | None) -> "TemplateInfo":
        desc = desc or {}
        return cls(
            label=desc.get("chat_template"),
            text=desc.get("chat_template_text"),
            source=desc.get("chat_template_source") or "none",
            sha256=desc.get("chat_template_sha256"),
            variant=desc.get("chat_template_variant"),
            builtin=desc.get("chat_template_builtin"),
            bos_token=desc.get("bos_token"),
            eos_token=desc.get("eos_token"),
            add_bos_token=desc.get("add_bos_token"),
        )


def template_from_text(
    text: str,
    *,
    source: str,
    label: str | None = None,
    variant: str | None = None,
    bos_token: str | None = None,
    eos_token: str | None = None,
    add_bos_token: bool | None = None,
) -> TemplateInfo:
    clean = text.strip()
    return TemplateInfo(
        label=label,
        text=clean or None,
        source=source,
        sha256=sha256_text(clean) if clean else None,
        variant=variant,
        bos_token=bos_token,
        eos_token=eos_token,
        add_bos_token=add_bos_token,
    )


def load_local_template(path: str | Path, *, label: str | None = None) -> TemplateInfo | None:
    """Load the first local template in the documented priority order."""
    root = Path(path)
    if not root.is_dir():
        return None

    tok = _read_json(root / "tokenizer_config.json")
    bos = _token_string(tok.get("bos_token"))
    eos = _token_string(tok.get("eos_token"))
    add_bos = tok.get("add_bos_token")
    add_bos = add_bos if isinstance(add_bos, bool) else None

    jinja = root / "chat_template.jinja"
    if jinja.is_file():
        try:
            text = jinja.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if text.strip():
            return template_from_text(
                text,
                source=f"local:{jinja.name}",
                label=label,
                bos_token=bos,
                eos_token=eos,
                add_bos_token=add_bos,
            )

    if tok.get("chat_template"):
        text, variant = select_variant(tok["chat_template"])
        if text:
            return template_from_text(
                text,
                source="local:tokenizer_config.json",
                label=label,
                variant=variant,
                bos_token=bos,
                eos_token=eos,
                add_bos_token=add_bos,
            )

    ctj = _read_json(root / "chat_template.json")
    if ctj:
        value = ctj.get("chat_template", ctj)
        text, variant = select_variant(value)
        if text:
            return template_from_text(
                text,
                source="local:chat_template.json",
                label=label,
                variant=variant,
                bos_token=bos,
                eos_token=eos,
                add_bos_token=add_bos,
            )

    cfg = _read_json(root / "config.json")
    if cfg.get("chat_template"):
        text, variant = select_variant(cfg["chat_template"])
        if text:
            return template_from_text(
                text,
                source="local:config.json",
                label=label,
                variant=variant,
                bos_token=bos,
                eos_token=eos,
                add_bos_token=add_bos,
            )
    return None


def apply_override(
    value: str | None = None,
    file_path: str | Path | None = None,
    *,
    label: str | None = None,
) -> TemplateInfo | None:
    """Apply a CLI template override.

    ``value`` is either Jinja text (anything containing ``{`` or a newline)
    or a llama.cpp built-in template name.  ``file_path`` reads Jinja text.
    """
    if value and file_path:
        raise ValueError("use --chat-template or --chat-template-file, not both")
    if file_path:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Chat template file not found: {path}")
        return template_from_text(
            path.read_text(encoding="utf-8"),
            source=f"cli:{path.name}",
            label=label,
        )
    if value is None:
        return None
    text = str(value)
    if not text.strip():
        raise ValueError("--chat-template cannot be empty")
    if "{" in text or "\n" in text:
        return template_from_text(text, source="cli:inline", label=label)
    name = text.strip()
    return TemplateInfo(
        label=label or name,
        source=f"cli:builtin:{name}",
        sha256=sha256_text(name),
        builtin=name,
    )


def write_template_file(path: str | Path, template: TemplateInfo) -> Path | None:
    """Write the resolved text into a step directory for reproducibility."""
    if not template.text:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.text, encoding="utf-8")
    return out


def document_messages(doc: dict[str, Any]) -> list[dict[str, str]]:
    """Turn a corpus bank document into an OpenAI-style message list."""
    turns = doc.get("turns")
    messages: list[dict[str, str]] = []
    if turns:
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "user").strip().lower()
            if role == "model":
                role = "assistant"
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(turn.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
    else:
        user = str(doc.get("user") or "").strip()
        if user:
            messages.append({"role": "user", "content": user})
        assistant = doc.get("assistant")
        if assistant is not None:
            content = str(assistant).strip()
            if content:
                messages.append({"role": "assistant", "content": content})
    if not messages:
        raise ValueError("Corpus document has no non-empty chat turns")
    return messages


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaTemplateRenderer:
    """Render messages with the model's template through ``llama-server``."""

    def __init__(
        self,
        *,
        model_gguf: str | Path,
        template: TemplateInfo,
        template_path: str | Path | None = None,
        log_path: str | Path | None = None,
        server_binary: str | Path | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model_gguf = Path(model_gguf).expanduser()
        self.template = template
        self.template_path = Path(template_path).expanduser() if template_path else None
        self.log_path = Path(log_path).expanduser() if log_path else None
        self.server_binary = server_binary
        self.timeout = float(timeout)
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self._url: str | None = None
        self._log_handle = None

    def _command(self) -> list[str]:
        binary = find_llama_binary("llama-server", self.server_binary)
        if binary is None:
            raise RuntimeError(
                "Step 07 needs llama-server to render the model's chat "
                "template. Install llama.cpp or set LLAMA_CPP_DIR."
            )
        if self.port is None:
            self.port = _free_port()
        cmd = [
            str(binary),
            "-m", str(self.model_gguf),
            "-ngl", "0",
            "-c", "512",
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--no-webui",
            "--no-warmup",
            "--jinja",
            "--no-prefill-assistant",
        ]
        if self.template.builtin:
            cmd += ["--chat-template", self.template.builtin]
        else:
            if self.template_path is None:
                raise RuntimeError("Template text is present but no template file was written")
            if not self.template_path.is_file():
                raise FileNotFoundError(f"Chat template file not found: {self.template_path}")
            cmd += ["--chat-template-file", str(self.template_path)]
        return cmd

    def _log_tail(self, limit: int = 40) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return ""
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-limit:])

    def _fail(self, message: str) -> RuntimeError:
        detail = self._log_tail()
        if detail:
            message = f"{message}\nllama-server log:\n{detail}"
        return RuntimeError(message)

    def __enter__(self) -> "LlamaTemplateRenderer":
        if not self.model_gguf.is_file():
            raise FileNotFoundError(f"Model GGUF not found: {self.model_gguf}")
        if not self.template.usable:
            raise RuntimeError(
                "No chat template is available. Step 07 will not guess a "
                "format; resolve the model's template or pass an override."
            )
        cmd = self._command()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("w", encoding="utf-8")
            stdout = self._log_handle
        else:
            stdout = subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            raise RuntimeError(f"Could not start llama-server: {exc}") from exc

        self._url = f"http://127.0.0.1:{self.port}/apply-template"
        health = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                message = self._fail(
                    f"llama-server exited with code {self.proc.returncode} before becoming ready"
                )
                self.__exit__(None, None, None)
                raise message
            try:
                with urllib.request.urlopen(health, timeout=2):
                    return self
            except Exception:  # noqa: BLE001 — 503 while loading is normal
                time.sleep(0.2)
        self.__exit__(None, None, None)
        raise self._fail(
            f"llama-server did not become ready within {self.timeout:.0f}s"
        )

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        if self._url is None:
            raise RuntimeError("Renderer is not started")
        payload = {
            "messages": messages,
            "add_generation_prompt": False,
        }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise self._fail(
                f"llama-server rejected the chat template (HTTP {exc.code}): {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise self._fail(f"Could not reach llama-server: {exc}") from exc
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            raise self._fail(f"llama-server returned no prompt: {data!r}")
        return prompt

    def render_document(self, doc: dict[str, Any]) -> str:
        return self.render_messages(document_messages(doc))

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
