"""
Structured logging and tracing for EB-RAG.

Uses structlog for structured logging with request_id propagation.
Integrates with OpenTelemetry for distributed tracing.
"""

import contextvars
import logging
import sys
import uuid
from typing import Any

import structlog
from structlog.typing import EventDict, WrappedLogger

from ebrag.common.config import LogLevel, get_settings

# Context variables for request tracking
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
tenant_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "tenant_id", default=""
)
mode_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mode", default=""
)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def set_request_context(
    request_id: str | None = None,
    tenant_id: str | None = None,
    mode: str | None = None,
) -> str:
    """Set request context for logging. Returns the request_id."""
    rid = request_id or generate_request_id()
    request_id_var.set(rid)
    if tenant_id:
        tenant_id_var.set(tenant_id)
    if mode:
        mode_var.set(mode)
    return rid


def clear_request_context() -> None:
    """Clear request context."""
    request_id_var.set("")
    tenant_id_var.set("")
    mode_var.set("")


def add_request_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Structlog processor to add request context to log entries."""
    request_id = request_id_var.get()
    tenant_id = tenant_id_var.get()
    mode = mode_var.get()

    if request_id:
        event_dict["request_id"] = request_id
    if tenant_id:
        event_dict["tenant_id"] = tenant_id
    if mode:
        event_dict["mode"] = mode

    return event_dict


def add_module_info(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add module and function information to log entries."""
    record = event_dict.get("_record")
    if record:
        event_dict["module"] = record.module
        event_dict["function"] = record.funcName
        event_dict["line"] = record.lineno
    return event_dict


def configure_logging() -> None:
    """Configure structlog for the application."""
    settings = get_settings()

    # Determine log level
    log_level = getattr(logging, settings.logging.level.value)

    # Shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_request_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.logging.format == "json":
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Set third-party loggers to WARNING
    for logger_name in ["httpx", "httpcore", "openai", "anthropic", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LogContext:
    """Context manager for logging with automatic context cleanup."""

    def __init__(
        self,
        request_id: str | None = None,
        tenant_id: str | None = None,
        mode: str | None = None,
        **extra: Any,
    ):
        self.request_id = request_id
        self.tenant_id = tenant_id
        self.mode = mode
        self.extra = extra
        self._tokens: list[contextvars.Token[str]] = []

    def __enter__(self) -> str:
        rid = set_request_context(
            request_id=self.request_id,
            tenant_id=self.tenant_id,
            mode=self.mode,
        )
        return rid

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        clear_request_context()


# Convenience functions for common log patterns
def log_request_start(
    logger: structlog.stdlib.BoundLogger,
    endpoint: str,
    **kwargs: Any,
) -> None:
    """Log the start of a request."""
    logger.info("request_started", endpoint=endpoint, **kwargs)


def log_request_end(
    logger: structlog.stdlib.BoundLogger,
    endpoint: str,
    duration_ms: float,
    status: str = "success",
    **kwargs: Any,
) -> None:
    """Log the end of a request."""
    logger.info(
        "request_completed",
        endpoint=endpoint,
        duration_ms=round(duration_ms, 2),
        status=status,
        **kwargs,
    )


def log_retrieval(
    logger: structlog.stdlib.BoundLogger,
    query: str,
    num_results: int,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Log retrieval operation."""
    logger.info(
        "retrieval_completed",
        query_length=len(query),
        num_results=num_results,
        duration_ms=round(duration_ms, 2),
        **kwargs,
    )


def log_dialectic_step(
    logger: structlog.stdlib.BoundLogger,
    step: str,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Log a dialectic engine step."""
    logger.info(
        "dialectic_step",
        step=step,
        duration_ms=round(duration_ms, 2),
        **kwargs,
    )


def log_llm_call(
    logger: structlog.stdlib.BoundLogger,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Log an LLM API call."""
    logger.info(
        "llm_call",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        duration_ms=round(duration_ms, 2),
        **kwargs,
    )


def log_citation_validation(
    logger: structlog.stdlib.BoundLogger,
    total: int,
    passed: int,
    failed: int,
    **kwargs: Any,
) -> None:
    """Log citation validation results."""
    logger.info(
        "citation_validation",
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=round(passed / total, 3) if total > 0 else 0.0,
        **kwargs,
    )
