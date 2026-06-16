"""
Chat and generation models.

Defines types for LLM interactions and generation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"  # OpenAI-compatible (e.g. Ollama Cloud)
    LOCAL = "local"


class MessageRole(str, Enum):
    """Message roles in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in conversation."""

    role: MessageRole
    content: str
    name: str | None = None  # For tool messages
    tool_call_id: str | None = None  # For tool responses


class ToolDefinition(BaseModel):
    """Definition of a callable tool."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_call_id: str
    name: str
    result: str
    error: str | None = None
    success: bool = True


class GenerationRequest(BaseModel):
    """Request for text generation."""

    messages: list[Message]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    stop_sequences: list[str] = Field(default_factory=list)

    # Tool use
    tools: list[ToolDefinition] = Field(default_factory=list)
    tool_choice: str | None = None  # auto, none, or specific tool name

    # Metadata
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    """Response from text generation."""

    content: str
    model: str
    finish_reason: str  # stop, length, tool_calls

    # Tool calls if any
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # Usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Timing
    generation_time_ms: float = 0.0

    # Metadata
    request_id: str | None = None


class StreamChunk(BaseModel):
    """A chunk in streaming generation."""

    content: str = ""
    finish_reason: str | None = None
    tool_call: ToolCall | None = None
    is_final: bool = False


class RAGContext(BaseModel):
    """Context prepared for RAG generation."""

    query: str
    passages: list[dict[str, Any]] = Field(default_factory=list)

    # Dialectic analysis
    has_conflicts: bool = False
    conflict_summary: str | None = None
    thesis_passages: list[str] = Field(default_factory=list)
    antithesis_passages: list[str] = Field(default_factory=list)

    # Synthesis guidance
    synthesis_prompt: str | None = None
    hedging_required: bool = False

    # Session context
    conversation_history: list[Message] = Field(default_factory=list)


class RAGResponse(BaseModel):
    """Complete RAG response with all metadata."""

    response_id: str
    query: str
    answer: str

    # Generation details
    model: str
    generation_time_ms: float = 0.0
    tokens_used: int = 0

    # Citations extracted
    citations: list[dict[str, Any]] = Field(default_factory=list)

    # Confidence
    confidence: float = 0.0
    retrieval_confidence: float = 0.0
    generation_confidence: float = 0.0
    attribution_confidence: float = 0.0

    # Dialectic
    has_conflicts: bool = False
    conflict_count: int = 0

    # Compliance
    compliance_passed: bool = True
    was_redacted: bool = False

    # Audit
    audit_id: str | None = None
    trace_id: str | None = None
