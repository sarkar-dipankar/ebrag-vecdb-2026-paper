"""
LLM client abstraction layer.

Provides unified interface for OpenAI, Anthropic, and local models.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ebrag.common import get_logger, get_settings
from ebrag.chat.models import (
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    Message,
    MessageRole,
    StreamChunk,
    ToolCall,
    ToolDefinition,
)

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response."""
        pass

    @abstractmethod
    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4-turbo-preview",
        base_url: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self.api_key = api_key or self.settings.llm.openai_api_key
        self.model = model
        self.base_url = base_url

        self._client: Any = None

        logger.info("openai_client_created", model=model)

    def _get_client(self) -> Any:
        """Lazy initialize OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for OpenAI API."""
        formatted = []
        for msg in messages:
            m: dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            formatted.append(m)
        return formatted

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tools for OpenAI API."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": tool.required,
                    },
                },
            }
            for tool in tools
        ]

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response using OpenAI."""
        client = self._get_client()
        start_time = time.perf_counter()

        model = request.model or self.model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }

        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        if request.tools:
            kwargs["tools"] = self._format_tools(request.tools)
            if request.tool_choice:
                kwargs["tool_choice"] = request.tool_choice

        response = await client.chat.completions.create(**kwargs)

        generation_time = (time.perf_counter() - start_time) * 1000

        # Extract tool calls if any
        tool_calls = []
        choice = response.choices[0]
        if choice.message.tool_calls:
            import json
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return GenerationResponse(
            content=choice.message.content or "",
            model=model,
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            generation_time_ms=generation_time,
            request_id=request.request_id,
        )

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response using OpenAI."""
        client = self._get_client()

        model = request.model or self.model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": True,
        }

        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        async for chunk in await client.chat.completions.create(**kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(
                    content=chunk.choices[0].delta.content,
                    finish_reason=chunk.choices[0].finish_reason,
                )

            if chunk.choices and chunk.choices[0].finish_reason:
                yield StreamChunk(
                    finish_reason=chunk.choices[0].finish_reason,
                    is_final=True,
                )

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-sonnet-20240229",
    ) -> None:
        self.settings = get_settings()
        self.api_key = api_key or self.settings.llm.anthropic_api_key
        self.model = model

        self._client: Any = None

        logger.info("anthropic_client_created", model=model)

    def _get_client(self) -> Any:
        """Lazy initialize Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    def _format_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Format messages for Anthropic API, extracting system prompt."""
        system_prompt = None
        formatted = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            else:
                formatted.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        return system_prompt, formatted

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tools for Anthropic API."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": tool.required,
                },
            }
            for tool in tools
        ]

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response using Anthropic."""
        client = self._get_client()
        start_time = time.perf_counter()

        model = request.model or self.model
        system_prompt, messages = self._format_messages(request.messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        if request.tools:
            kwargs["tools"] = self._format_tools(request.tools)

        response = await client.messages.create(**kwargs)

        generation_time = (time.perf_counter() - start_time) * 1000

        # Extract content
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return GenerationResponse(
            content=content,
            model=model,
            finish_reason=response.stop_reason or "end_turn",
            tool_calls=tool_calls,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            generation_time_ms=generation_time,
            request_id=request.request_id,
        )

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response using Anthropic."""
        client = self._get_client()

        model = request.model or self.model
        system_prompt, messages = self._format_messages(request.messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(content=text)

            yield StreamChunk(
                finish_reason="end_turn",
                is_final=True,
            )

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model


class LLMClientFactory:
    """Factory for creating LLM clients."""

    _clients: dict[str, BaseLLMClient] = {}

    @classmethod
    def get_client(
        cls,
        provider: LLMProvider | str = LLMProvider.OPENAI,
        model: str | None = None,
        **kwargs: Any,
    ) -> BaseLLMClient:
        """Get or create an LLM client."""
        if isinstance(provider, str):
            provider = LLMProvider(provider)

        # Create key for caching
        key = f"{provider.value}:{model or 'default'}"

        if key not in cls._clients:
            # OpenAI-compatible providers (OpenAI, Ollama Cloud, local OpenAI servers)
            # all use OpenAIClient; pull base_url/api_key from settings when not given.
            if provider in (
                LLMProvider.OPENAI,
                LLMProvider.OLLAMA,
                LLMProvider.LOCAL,
            ):
                from ebrag.common import get_settings

                client_kwargs = get_settings().llm.openai_client_kwargs()
                for k, v in client_kwargs.items():
                    kwargs.setdefault(k, v)
                cls._clients[key] = OpenAIClient(
                    model=model or "gpt-4-turbo-preview",
                    **kwargs,
                )
            elif provider == LLMProvider.ANTHROPIC:
                cls._clients[key] = AnthropicClient(
                    model=model or "claude-3-sonnet-20240229",
                    **kwargs,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        return cls._clients[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the client cache."""
        cls._clients.clear()


# Convenience functions
def get_llm_client(
    provider: LLMProvider | str = LLMProvider.OPENAI,
    model: str | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """Get an LLM client."""
    return LLMClientFactory.get_client(provider, model, **kwargs)
