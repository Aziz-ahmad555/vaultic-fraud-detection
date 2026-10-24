"""Language models behind the copilot (D224): one small interface, two implementations.

  OllamaLLM    HTTP to a local Ollama server on the loopback interface only (127.0.0.1:11434 by
               default, VAULTIC_OLLAMA_HOST to change the port); any other host is refused, so no
               case evidence can leave the laptop. Streams tokens from /api/chat.
  ScriptedLLM  deterministic answers for tests and CI (CI never needs Ollama): a fixed text or a
               function of the messages; it records every call.

Both raise LLMUnavailable with a message an analyst can act on when the model cannot answer.
Standard library only.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Protocol
from urllib.parse import urlparse

DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"  # D223
DEFAULT_HOST = "http://127.0.0.1:11434"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
START_HINT = "start it with E:\\ollama\\start_ollama.cmd (Ollama listening on 127.0.0.1:11434)"


class LLMUnavailable(RuntimeError):
    """The model cannot answer (server not running, model not pulled, broken stream)."""


class LLM(Protocol):
    model: str

    def check(self) -> None:
        """Raise LLMUnavailable if the model cannot answer now."""

    def stream(self, messages: list[dict], max_tokens: int) -> Iterator[str]:
        """Yield the answer's text pieces in order."""


def loopback_host(host: str) -> str:
    """The host URL if it points at this machine's loopback interface; ValueError otherwise."""
    u = urlparse(host)
    if u.scheme != "http" or u.hostname not in LOOPBACK:
        raise ValueError(
            f"the copilot only talks to a local model on the loopback interface, not {host!r}"
        )
    return host.rstrip("/")


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 600):
        self.model = model or os.environ.get("VAULTIC_COPILOT_MODEL") or DEFAULT_MODEL
        self.host = loopback_host(host or os.environ.get("VAULTIC_OLLAMA_HOST") or DEFAULT_HOST)
        self.timeout = timeout

    def _open(self, path: str, body: dict | None = None, timeout: float | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"})  # fmt: skip
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def check(self) -> None:
        try:
            with self._open("/api/tags", timeout=5) as r:
                names = {m.get("name") for m in json.loads(r.read()).get("models", [])}
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise LLMUnavailable(f"The local language model is not running: {START_HINT}.") from e
        if self.model not in names:
            raise LLMUnavailable(f"The model {self.model} is not installed in Ollama "
                                 f"(installed: {', '.join(sorted(n for n in names if n)) or 'none'}).")  # fmt: skip

    def stream(self, messages: list[dict], max_tokens: int) -> Iterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True,
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096}}  # fmt: skip
        try:
            with self._open("/api/chat", body) as r:
                for line in r:
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg.get("error"):
                        raise LLMUnavailable(f"The local language model failed: {msg['error']}")
                    piece = msg.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if msg.get("done"):
                        return
        except urllib.error.HTTPError as e:
            raise LLMUnavailable(
                f"The local language model refused the request (HTTP {e.code})."
            ) from e
        except (urllib.error.URLError, OSError) as e:
            raise LLMUnavailable(f"The local language model is not running: {START_HINT}.") from e


class ScriptedLLM:
    """answer: a fixed text, or a function (messages) -> text. available=False behaves like a
    model that is not running. The text is streamed word by word."""

    def __init__(self, answer: str | Callable[[list[dict]], str] = "", model: str = "scripted",
                 available: bool = True):  # fmt: skip
        self.answer, self.model, self.available = answer, model, available
        self.calls: list[dict] = []

    def check(self) -> None:
        if not self.available:
            raise LLMUnavailable(f"The local language model is not running: {START_HINT}.")

    def stream(self, messages: list[dict], max_tokens: int) -> Iterator[str]:
        self.check()
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        text = self.answer(messages) if callable(self.answer) else self.answer
        words = text.split(" ")
        for i, w in enumerate(words):
            yield w if i == len(words) - 1 else w + " "
