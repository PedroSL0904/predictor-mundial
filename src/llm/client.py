"""Cliente para la API de OpenCode (compatible con Anthropic Messages).

Modelo por defecto: minimax-m3 (minimax m3)
Endpoint: https://opencode.ai/zen/go/v1/messages

Configuracion:
    API key via variable de entorno OPENCODE_API_KEY
    O archivo .env (gitignored) en raiz del proyecto
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "minimax-m3"
DEFAULT_TIMEOUT = 60


def _load_api_key() -> str:
    """Carga API key de env var o .env file."""
    key = os.environ.get("OPENCODE_API_KEY")
    if key:
        return key
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENCODE_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "OPENCODE_API_KEY no encontrada. Configura la variable de entorno "
        "o crea un archivo .env con OPENCODE_API_KEY=sk-..."
    )


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost: str | None = None
    model: str = ""


class OpenCodeClient:
    """Cliente para la API de OpenCode (formato Anthropic Messages)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or _load_api_key()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def messages(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Llama al endpoint /v1/messages.

        Args:
            prompt: mensaje del usuario.
            system: prompt de sistema opcional.
            max_tokens: limite de tokens de salida.
            temperature: 0.0 = deterministico, 1.0 = creativo.
        """
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        messages = [{"role": "user", "content": prompt}]
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        # Extraer texto de content blocks
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cost=data.get("cost"),
            model=data.get("model", self.model),
        )
