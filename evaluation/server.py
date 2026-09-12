"""Talk to a local llama-server. No Hugging Face, no Python model bindings."""

from __future__ import annotations

import json
import math
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from llama_bins import find_llama_binary


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _post(url: str, payload: dict[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get(url: str, *, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except urllib.error.URLError:
        return 0, ""


class LlamaServer:
    """One GGUF, one server process, greedy decoding + next-token logprobs."""

    def __init__(
        self,
        gguf: Path,
        *,
        n_gpu_layers: int = 99,
        ctx: int = 4096,
        port: int | None = None,
    ) -> None:
        self.gguf = Path(gguf)
        self.port = port or _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self._proc: subprocess.Popen[str] | None = None
        binary = find_llama_binary("llama-server")
        if binary is None:
            raise FileNotFoundError(
                "llama-server not found. Set LLAMA_CPP_DIR to a llama.cpp build."
            )
        if not self.gguf.is_file():
            raise FileNotFoundError(f"GGUF not found: {self.gguf}")
        self._cmd = [
            str(binary),
            "-m",
            str(self.gguf),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(ctx),
            "-ngl",
            str(n_gpu_layers),
        ]
        self.supports_loglikelihood = True

    def start(self, *, timeout_s: float = 120.0) -> None:
        self._proc = subprocess.Popen(
            self._cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() if self._proc.stderr else "")[-2000:]
                raise RuntimeError(f"llama-server exited early:\n{err}")
            code, _ = _get(f"{self.base}/health")
            if code == 200:
                return
            time.sleep(0.4)
        raise TimeoutError(f"llama-server did not become healthy on {self.base}")

    def close(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def tokenize(self, text: str, *, add_special: bool = False) -> list[int]:
        out = _post(
            f"{self.base}/tokenize",
            {"content": text, "add_special": add_special},
        )
        tokens = out.get("tokens") or []
        ids: list[int] = []
        for t in tokens:
            if isinstance(t, int):
                ids.append(t)
            elif isinstance(t, dict) and "id" in t:
                ids.append(int(t["id"]))
        return ids

    def completion(
        self,
        prompt: str | list[int],
        *,
        n_predict: int = 1,
        n_probs: int = 0,
        stop: list[str] | None = None,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "cache_prompt": True,
        }
        if n_probs:
            payload["n_probs"] = n_probs
        if stop:
            payload["stop"] = stop
        return _post(f"{self.base}/completion", payload, timeout=timeout)

    def next_logprobs(self, prompt: str | list[int], *, top: int = 128) -> dict[str, float]:
        """log P(token | prompt) for the next token, keyed by token string."""
        raw = self.completion(prompt, n_predict=1, n_probs=top)
        return _parse_next_logprobs(raw)

    def generate(self, prompt: str, *, n_predict: int = 256, stop: list[str] | None = None) -> str:
        raw = self.completion(prompt, n_predict=n_predict, stop=stop)
        return str(raw.get("content") or "")

    def loglikelihood(self, context: str, continuation: str) -> tuple[float, int]:
        """Sum of next-token logprobs of ``continuation`` given ``context``."""
        ctx_ids = self.tokenize(context, add_special=True)
        cont_ids = self.tokenize(continuation, add_special=False)
        if not cont_ids:
            return 0.0, 0
        total = 0.0
        prefix = list(ctx_ids)
        for tid in cont_ids:
            dist = self.next_logprobs(prefix, top=128)
            # Match by token id string if present, else skip (counts as very low).
            logp = dist.get(f"id:{tid}")
            if logp is None:
                # Token not in the returned top-k. min(top-k) would overstate
                # probability; a large negative keeps ranking conservative.
                logp = -99.0
            total += float(logp)
            prefix.append(tid)
        return total, len(cont_ids)


def _parse_next_logprobs(raw: dict[str, Any]) -> dict[str, float]:
    """Normalize llama-server probability blobs across versions."""
    out: dict[str, float] = {}
    blobs = (
        raw.get("completion_probabilities")
        or raw.get("probs")
        or raw.get("completion_probs")
        or []
    )
    if not blobs and isinstance(raw.get("tokens"), list):
        blobs = raw["tokens"]
    first = blobs[0] if blobs else {}
    items = []
    if isinstance(first, dict):
        items = first.get("probs") or first.get("top_logprobs") or first.get("top_probs") or []
        if not items and "tok_str" in first:
            items = blobs
    for item in items:
        if not isinstance(item, dict):
            continue
        tok = item.get("tok_str") or item.get("token") or item.get("content") or ""
        tid = item.get("id") or item.get("token_id")
        if "logprob" in item:
            lp = float(item["logprob"])
        elif "prob" in item:
            p = float(item["prob"])
            lp = float("-inf") if p <= 0 else math.log(p)
        else:
            continue
        if tok:
            out[str(tok)] = lp
            out[str(tok).strip()] = lp
        if tid is not None:
            out[f"id:{int(tid)}"] = lp
    return out


def letter_logprobs(dist: dict[str, float], letters: tuple[str, ...] = ("A", "B", "C", "D")) -> dict[str, float]:
    """Collapse tokenizer variants ('A', ' A', 'A\\n') onto choice letters."""
    scored = {L: float("-inf") for L in letters}
    for key, lp in dist.items():
        if key.startswith("id:"):
            continue
        text = str(key).strip().strip(".)")
        if text in scored and lp > scored[text]:
            scored[text] = lp
    return scored


OLLAMA_TAGS = {
    "functiongemma-270m-bf16.gguf": "functiongemma-src",
    "original.gguf": "functiongemma-src",
    "functiongemma-270m-q4_k_m.gguf": "functiongemma-q4",
    "q4_k_m.gguf": "functiongemma-q4",
    "functiongemma-270m-odg.gguf": "functiongemma-odg",
    "opendynamic.gguf": "functiongemma-odg",
}


def ollama_tag_for(gguf: Path) -> str | None:
    return OLLAMA_TAGS.get(Path(gguf).name) or OLLAMA_TAGS.get(Path(gguf).name.lower())


class OllamaServer:
    """Greedy generate via local Ollama. No token logprobs."""

    supports_loglikelihood = False

    def __init__(self, tag: str, *, host: str = "http://127.0.0.1:11434") -> None:
        self.tag = tag
        self.base = host.rstrip("/")
        self.gguf = Path(tag)

    def start(self) -> None:
        code, _ = _get(f"{self.base}/api/tags")
        if code != 200:
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")

    def close(self) -> None:
        return None

    def __enter__(self) -> "OllamaServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def generate(self, prompt: str, *, n_predict: int = 256, stop: list[str] | None = None) -> str:
        raw = _post(
            f"{self.base}/api/generate",
            {
                "model": self.tag,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": n_predict,
                    "stop": stop or [],
                },
            },
            timeout=180.0,
        )
        if raw.get("error"):
            raise RuntimeError(f"Ollama generate failed: {raw['error']}")
        return str(raw.get("response") or "")

    def loglikelihood(self, context: str, continuation: str) -> tuple[float, int]:
        raise NotImplementedError("Ollama backend has no continuation logprobs")

    def next_logprobs(self, prompt: str | list[int], *, top: int = 128) -> dict[str, float]:
        raise NotImplementedError("Ollama backend has no next-token logprobs")


def open_backend(gguf: Path):
    """llama-server if it can load the GGUF; otherwise a matching Ollama tag."""
    server: LlamaServer | None = None
    try:
        server = LlamaServer(gguf)
        server.start()
        return server
    except Exception as exc:
        if server is not None:
            server.close()
        tag = ollama_tag_for(gguf)
        if tag is None:
            raise
        note = (
            f"llama-server could not load {Path(gguf).name} "
            f"({exc.__class__.__name__}); using Ollama tag {tag}"
        )
        backend = OllamaServer(tag)
        backend.start()
        backend._fallback_note = note  # noqa: SLF001
        return backend
