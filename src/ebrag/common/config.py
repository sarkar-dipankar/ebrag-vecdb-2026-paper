"""
Configuration schema for EB-RAG.

Uses pydantic-settings for environment variable loading and validation.
All settings can be overridden via environment variables with EBRAG_ prefix.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load a local .env (if present) so the EBRAG_* / provider keys documented in
# .env.example actually populate the environment before settings are built.
# Existing environment variables take precedence (override=False).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is a declared dependency; degrade gracefully.
    pass


class Mode(str, Enum):
    """Pipeline execution mode."""

    VANILLA = "vanilla"
    EBRAG = "eb-rag"
    BENCHMARK = "benchmark"


class LogLevel(str, Enum):
    """Logging level."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # OpenAI-compatible (e.g. Ollama Cloud); set base_url


class VectorStoreType(str, Enum):
    """Supported vector store backends."""

    USEARCH = "usearch"
    VESPA = "vespa"


class SparseIndexType(str, Enum):
    """Supported sparse index backends."""

    BM25 = "bm25"
    SPLADE = "splade"


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="EBRAG_LOG_")

    level: LogLevel = LogLevel.INFO
    format: Literal["json", "console"] = "json"
    include_trace_id: bool = True


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="EBRAG_LLM_")

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: float = Field(default=60.0, gt=0)

    # Model tiers for cost optimization
    critic_model: str | None = None  # Falls back to main model
    synthesizer_model: str | None = None  # Falls back to main model

    # Base URL for OpenAI-compatible endpoints (Ollama Cloud, vLLM, proxies).
    # Maps to EBRAG_LLM_BASE_URL.
    base_url: str | None = None

    # API keys loaded from environment
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    ollama_api_key: str | None = Field(default=None, alias="OLLAMA_API_KEY")

    def openai_client_kwargs(self) -> dict[str, Any]:
        """Kwargs for an OpenAI-compatible client.

        Works for both native OpenAI and OpenAI-compatible backends such as Ollama
        Cloud: when ``provider`` is ``ollama`` (or a ``base_url`` is set) we pass the
        ``base_url`` and the matching key. Used by the benchmark runner and the
        synthesis engine, which talk to the OpenAI SDK directly.
        """
        kwargs: dict[str, Any] = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.provider == LLMProvider.OLLAMA:
            # Ollama Cloud requires a non-empty key alongside the base_url.
            kwargs["api_key"] = self.ollama_api_key or self.openai_api_key or "ollama"
        elif self.openai_api_key:
            kwargs["api_key"] = self.openai_api_key
        return kwargs


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="EBRAG_EMBEDDING_")

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True


class RetrievalSettings(BaseSettings):
    """Retrieval layer configuration."""

    model_config = SettingsConfigDict(env_prefix="EBRAG_RETRIEVAL_")

    vector_store: VectorStoreType = VectorStoreType.USEARCH
    sparse_index: SparseIndexType = SparseIndexType.BM25

    # Retrieval parameters
    top_k: int = Field(default=10, gt=0)
    thesis_k: int = Field(default=5, gt=0)
    antithesis_k: int = Field(default=5, gt=0)

    # Diversity scoring
    diversity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Cross-encoder rescoring
    use_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Performance
    cache_embeddings: bool = True
    index_path: Path = Path("data/indices")


class DialecticSettings(BaseSettings):
    """Dialectic engine configuration."""

    model_config = SettingsConfigDict(env_prefix="EBRAG_DIALECTIC_")

    # NLI model for conflict detection
    nli_model: str = "microsoft/deberta-base-mnli"
    contradiction_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    # Synthesis model
    synthesis_model: str = "gpt-4o-mini"

    # Drafting
    parallel_drafts: bool = True
    draft_max_tokens: int = 1024

    # Critic
    critic_severity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    escalate_to_human_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    # Synthesis
    synthesis_max_tokens: int = 2048

    # Calibration
    calibration_method: Literal["platt", "temperature"] = "temperature"
    calibration_bins: int = Field(default=10, gt=0)

    # Policy-based routing
    skip_dialectic_threshold: float = Field(default=0.2, ge=0.0, le=1.0)


class ComplianceSettings(BaseSettings):
    """Quality and compliance configuration."""

    model_config = SettingsConfigDict(env_prefix="EBRAG_COMPLIANCE_")

    # Citation validation
    validate_citations: bool = True
    citation_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    block_on_citation_failure: bool = True

    # Policy
    policy_packs: list[str] = Field(default_factory=list)
    redact_pii: bool = True

    # Audit
    audit_enabled: bool = True
    audit_storage_path: Path = Path("data/audit")
    audit_retention_days: int = Field(default=365, gt=0)


class ServingSettings(BaseSettings):
    """API serving configuration."""

    model_config = SettingsConfigDict(env_prefix="EBRAG_SERVING_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # Rate limiting
    rate_limit_requests: int = Field(default=100, gt=0)
    rate_limit_window: int = Field(default=60, gt=0)
    benchmark_rate_multiplier: float = Field(default=10.0, gt=0)

    # Streaming
    enable_streaming: bool = True

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class ChatSettings(BaseSettings):
    """Chat mode configuration."""

    model_config = SettingsConfigDict(env_prefix="EBRAG_CHAT_")

    session_ttl_seconds: int = Field(default=3600, gt=0)
    max_history_turns: int = Field(default=20, gt=0)
    summary_threshold_tokens: int = Field(default=2000, gt=0)


class BenchmarkSettings(BaseSettings):
    """Benchmark harness configuration."""

    model_config = SettingsConfigDict(env_prefix="EBRAG_BENCHMARK_")

    datasets_path: Path = Path("data/benchmarks")
    traces_path: Path = Path("benchmarks/traces")
    reports_path: Path = Path("benchmarks/reports")

    # Regression gates
    min_accuracy_uplift: float = Field(default=0.08, ge=0.0)  # 8%
    max_contradiction_rate: float = Field(default=0.70, ge=0.0, le=1.0)  # 30% reduction = 70% of baseline
    max_ece_drift: float = Field(default=0.05, ge=0.0)  # 5%
    max_chat_em_drop: float = Field(default=0.05, ge=0.0)  # 5%

    # Cost guardrails
    max_cost_increase_ratio: float = Field(default=1.35, ge=1.0)  # 35% max increase


class Settings(BaseSettings):
    """Root configuration combining all settings."""

    model_config = SettingsConfigDict(
        env_prefix="EBRAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Core
    mode: Mode = Mode.EBRAG
    debug: bool = False
    data_path: Path = Path("data")

    # Nested settings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    dialectic: DialecticSettings = Field(default_factory=DialecticSettings)
    compliance: ComplianceSettings = Field(default_factory=ComplianceSettings)
    serving: ServingSettings = Field(default_factory=ServingSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)

    @field_validator("data_path", mode="after")
    @classmethod
    def ensure_data_path_exists(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
