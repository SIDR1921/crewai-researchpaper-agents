"""One place for paths and model choices, so notebook and local runs never diverge.

Resolution order for the data root:
  1. ``LITREVIEW_ROOT`` environment variable
  2. ``/content/drive/MyDrive/litreview`` when running on Colab with Drive mounted
  3. the repository root (local development)

Secrets come from Colab ``userdata`` when available, otherwise from the
environment (``.env`` is loaded if present). Nothing is ever hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, local only
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COLAB_ROOT = Path("/content/drive/MyDrive/litreview")

if load_dotenv is not None:
    load_dotenv(_REPO_ROOT / ".env", override=False)


def on_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from Colab userdata, else the environment."""
    if on_colab():
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:
            pass
    return os.environ.get(name, default)


def data_root() -> Path:
    env = os.environ.get("LITREVIEW_ROOT")
    if env:
        return Path(env)
    if _COLAB_ROOT.exists():
        return _COLAB_ROOT
    return _REPO_ROOT


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    sources: Path
    results: Path
    figures: Path
    models: Path
    cache: Path

    def mkdirs(self) -> "Paths":
        for p in (self.data, self.sources, self.results, self.figures, self.models, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


def paths(root: str | Path | None = None) -> Paths:
    r = Path(root) if root else data_root()
    return Paths(
        root=r,
        data=r / "data",
        sources=r / "data" / "sources",
        results=r / "results",
        figures=r / "figures",
        models=r / "models",
        cache=r / "data" / "cache",
    )


# ---------------------------------------------------------------- model config

#: Provider presets. ``model`` strings are LiteLLM model strings, which is what
#: CrewAI's ``LLM`` class takes. Prices are USD per 1M tokens and are only used
#: for the cost column in the usage log; free tiers are recorded as 0.0 but the
#: token counts are still logged so a paid-tier cost can be computed later.
PROVIDERS: dict[str, dict] = {
    # Checked live against https://api.groq.com/openai/v1/models on 2026-10-20.
    # The Llama models the free tier used to serve are gone; the chat-capable
    # models now are qwen3.8-27b and the two gpt-oss sizes. The gpt-oss models are
    # reasoning models that spend their completion budget on hidden reasoning
    # tokens before emitting anything, which is expensive under an 8k-token/minute
    # cap, so the default is Qwen. Limits below are read from this account's own
    # x-ratelimit-* response headers, not from documentation.
    "groq": {
        "model": "groq/qwen/qwen3.8-27b",
        "small_model": "groq/openai/gpt-oss-20b",
        "api_key_env": "GROQ_API_KEY",
        "rpm": 20,
        "tpm": 7400,        # account limit is 8000/min; leave headroom for retries
        "price_in": 0.0,
        "price_out": 0.0,
    },
    "gemini": {
        "model": "gemini/gemini-2.0-flash",
        "small_model": "gemini/gemini-2.0-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
        "rpm": 14,
        "tpm": 900_000,
        "price_in": 0.0,
        "price_out": 0.0,
    },
    "ollama": {
        "model": "ollama/qwen2.5:7b",
        "small_model": "ollama/llama3.2:latest",
        "api_key_env": None,
        "rpm": 10_000,
        "tpm": 10_000_000,
        "price_in": 0.0,
        "price_out": 0.0,
    },
}

#: Hugging Face checkpoints. Pinned here so the report can state exact versions.
HF_MODELS = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "nli": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "embed": "sentence-transformers/all-MiniLM-L6-v2",
    "distill_base": "Qwen/Qwen2.5-1.5B-Instruct",
}


@dataclass
class Settings:
    provider: str = field(default_factory=lambda: os.environ.get("LITREVIEW_PROVIDER", "ollama"))
    model: str | None = field(default_factory=lambda: os.environ.get("LITREVIEW_MODEL"))
    small_model: str | None = field(default_factory=lambda: os.environ.get("LITREVIEW_SMALL_MODEL"))
    temperature: float = 0.1
    max_tokens: int = 2048
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    seed: int = 0

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}; pick one of {sorted(PROVIDERS)}")
        preset = PROVIDERS[self.provider]
        self.model = self.model or preset["model"]
        self.small_model = self.small_model or preset["small_model"]

    @property
    def preset(self) -> dict:
        return PROVIDERS[self.provider]

    @property
    def api_key(self) -> str | None:
        env = self.preset["api_key_env"]
        return secret(env) if env else None

    def describe(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "small_model": self.small_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "hf_models": HF_MODELS,
        }


SETTINGS = Settings()
