"""
Chat and generation module.

Provides:
- LLM client abstraction (OpenAI, Anthropic)
- RAG pipeline orchestration
- Prompt building and templates
- Tool/function calling support
"""

from ebrag.chat.llm import (
    AnthropicClient,
    BaseLLMClient,
    LLMClientFactory,
    OpenAIClient,
    get_llm_client,
)
from ebrag.chat.models import (
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    Message,
    MessageRole,
    RAGContext,
    RAGResponse,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from ebrag.chat.pipeline import RAGPipeline, get_rag_pipeline
from ebrag.chat.prompts import PromptBuilder, get_prompt_builder
from ebrag.chat.tools import (
    ToolExecutor,
    ToolRegistry,
    get_tool_registry,
    register_tool,
)

__all__ = [
    # LLM Clients
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "LLMClientFactory",
    "get_llm_client",
    # Models
    "LLMProvider",
    "MessageRole",
    "Message",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "GenerationRequest",
    "GenerationResponse",
    "StreamChunk",
    "RAGContext",
    "RAGResponse",
    # Pipeline
    "RAGPipeline",
    "get_rag_pipeline",
    # Prompts
    "PromptBuilder",
    "get_prompt_builder",
    # Tools
    "ToolRegistry",
    "ToolExecutor",
    "get_tool_registry",
    "register_tool",
]
