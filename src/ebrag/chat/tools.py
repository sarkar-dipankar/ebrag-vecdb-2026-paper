"""
Tool/function calling support.

Provides a registry and executor for LLM tool calls.
"""

import asyncio
import inspect
from typing import Any, Callable, Coroutine

from ebrag.common import get_logger
from ebrag.chat.models import (
    ToolCall,
    ToolDefinition,
    ToolResult,
)

logger = get_logger(__name__)


class ToolRegistry:
    """
    Registry for callable tools.

    Allows registration and execution of tools that can be
    called by the LLM.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._definitions: dict[str, ToolDefinition] = {}

        # Register built-in tools
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in tools."""
        # Knowledge base search
        self.register(
            name="search_knowledge_base",
            description="Search the knowledge base for relevant information",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "namespace": {"type": "string", "description": "Namespace to search in"},
                "top_k": {"type": "integer", "description": "Number of results"},
            },
            required=["query"],
            handler=self._search_knowledge_base,
        )

        # Get source details
        self.register(
            name="get_source_details",
            description="Get detailed information about a source document",
            parameters={
                "source_id": {"type": "string", "description": "Source document ID"},
            },
            required=["source_id"],
            handler=self._get_source_details,
        )

        # Verify claim
        self.register(
            name="verify_claim",
            description="Verify a specific claim against the knowledge base",
            parameters={
                "claim": {"type": "string", "description": "Claim to verify"},
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source IDs to check against",
                },
            },
            required=["claim"],
            handler=self._verify_claim,
        )

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        required: list[str],
        handler: Callable[..., Any],
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter definitions
            required: Required parameter names
            handler: Function to call
        """
        self._tools[name] = handler
        self._definitions[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            required=required,
        )

        logger.debug("tool_registered", name=name)

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            del self._definitions[name]
            return True
        return False

    def get_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions."""
        return list(self._definitions.values())

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Get a specific tool definition."""
        return self._definitions.get(name)

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: Tool call from LLM

        Returns:
            ToolResult with execution result
        """
        logger.info(
            "tool_execute",
            tool=tool_call.name,
            args=list(tool_call.arguments.keys()),
        )

        if tool_call.name not in self._tools:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result="",
                error=f"Tool '{tool_call.name}' not found",
                success=False,
            )

        handler = self._tools[tool_call.name]

        try:
            # Check if async
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**tool_call.arguments)
            else:
                result = handler(**tool_call.arguments)

            # Convert to string if needed
            if not isinstance(result, str):
                import json
                result = json.dumps(result, default=str)

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=result,
                success=True,
            )

        except Exception as e:
            logger.error(
                "tool_error",
                tool=tool_call.name,
                error=str(e),
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result="",
                error=str(e),
                success=False,
            )

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls."""
        results = []
        for call in tool_calls:
            result = await self.execute(call)
            results.append(result)
        return results

    # Built-in tool handlers

    async def _search_knowledge_base(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search the knowledge base."""
        try:
            from ebrag.retrieval import get_hybrid_retriever

            retriever = get_hybrid_retriever()
            result = await retriever.retrieve(
                query=query,
                namespace=namespace,
                top_k=top_k,
            )

            return {
                "query": query,
                "results": [
                    {
                        "passage_id": p.passage_id,
                        "source_id": p.source_id,
                        "text": p.text[:500],
                        "score": p.score,
                    }
                    for p in result.passages
                ],
                "total": len(result.passages),
            }
        except Exception as e:
            return {"error": str(e), "results": []}

    async def _get_source_details(self, source_id: str) -> dict[str, Any]:
        """Get source document details."""
        # Placeholder - would connect to document store
        return {
            "source_id": source_id,
            "title": f"Document {source_id}",
            "type": "unknown",
            "metadata": {},
        }

    async def _verify_claim(
        self,
        claim: str,
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Verify a claim against sources."""
        try:
            from ebrag.compliance import get_citation_validator

            validator = get_citation_validator()

            # Simplified verification
            return {
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "sources_checked": source_ids or [],
                "note": "Full verification requires provenance record",
            }
        except Exception as e:
            return {"claim": claim, "error": str(e), "verified": False}


class ToolExecutor:
    """
    Executes tools in a conversation loop.

    Handles multi-turn tool calling until completion.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        max_iterations: int = 10,
    ) -> None:
        self.registry = registry or get_tool_registry()
        self.max_iterations = max_iterations

    async def run_with_tools(
        self,
        llm_client: Any,
        messages: list[Any],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> tuple[str, list[ToolResult]]:
        """
        Run generation with tool calling loop.

        Args:
            llm_client: LLM client to use
            messages: Initial messages
            temperature: Generation temperature
            max_tokens: Max tokens

        Returns:
            (final_response, tool_results)
        """
        from ebrag.chat.models import GenerationRequest, Message, MessageRole

        all_tool_results: list[ToolResult] = []
        current_messages = list(messages)

        for iteration in range(self.max_iterations):
            request = GenerationRequest(
                messages=current_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=self.registry.get_definitions(),
                tool_choice="auto" if iteration < self.max_iterations - 1 else "none",
            )

            response = await llm_client.generate(request)

            # If no tool calls, we're done
            if not response.tool_calls:
                return response.content, all_tool_results

            # Execute tools
            tool_results = await self.registry.execute_all(response.tool_calls)
            all_tool_results.extend(tool_results)

            # Add assistant message with tool calls
            current_messages.append(Message(
                role=MessageRole.ASSISTANT,
                content=response.content or "",
            ))

            # Add tool results
            for result in tool_results:
                current_messages.append(Message(
                    role=MessageRole.TOOL,
                    content=result.result if result.success else f"Error: {result.error}",
                    name=result.name,
                    tool_call_id=result.tool_call_id,
                ))

        # Max iterations reached
        logger.warning("tool_loop_max_iterations")
        return current_messages[-1].content if current_messages else "", all_tool_results


# Global registry
_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to register a function as a tool.

    Usage:
        @register_tool(
            name="my_tool",
            description="Does something",
            parameters={"arg": {"type": "string"}},
        )
        async def my_tool(arg: str) -> str:
            return f"Result: {arg}"
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        registry = get_tool_registry()
        registry.register(
            name=name,
            description=description,
            parameters=parameters,
            required=required or [],
            handler=func,
        )
        return func

    return decorator
