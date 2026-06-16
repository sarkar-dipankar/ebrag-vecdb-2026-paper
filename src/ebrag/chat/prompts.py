"""
Prompt templates and builders for RAG generation.

Provides structured prompts for different RAG modes.
"""

from typing import Any

from ebrag.chat.models import Message, MessageRole, RAGContext


# --- System Prompts ---

SYSTEM_PROMPT_VANILLA = """You are a helpful assistant that answers questions based on the provided context.

Instructions:
- Answer based ONLY on the information in the provided passages
- If the context doesn't contain enough information, say so
- Be concise and direct
- Cite your sources using [1], [2], etc. notation"""

SYSTEM_PROMPT_EBRAG = """You are an evidence-balanced assistant that provides nuanced answers by considering multiple perspectives.

Instructions:
- Carefully analyze ALL provided passages for relevant information
- When sources agree, present the consensus view with confidence
- When sources CONFLICT or DISAGREE, you MUST:
  1. Acknowledge the disagreement explicitly
  2. Present both perspectives fairly
  3. Use hedging language (e.g., "according to Source A...", "however, Source B suggests...")
  4. Explain why the disagreement might exist if possible
- Always cite your sources using [1], [2], etc. notation
- Never fabricate information not in the passages
- If you're uncertain, express that uncertainty"""

SYSTEM_PROMPT_BENCHMARK = SYSTEM_PROMPT_EBRAG + """

Additional benchmark mode instructions:
- Be especially precise in your citations
- Structure your answer clearly with evidence for each claim
- Note any limitations in the available evidence"""


# --- Context Templates ---

CONTEXT_TEMPLATE = """
## Retrieved Passages

{passages}

## User Question
{query}
"""

CONTEXT_WITH_CONFLICT_TEMPLATE = """
## Retrieved Passages

{passages}

## Conflict Analysis
The retrieved passages contain conflicting information:
{conflict_summary}

**Thesis passages** (supporting one view): {thesis_ids}
**Antithesis passages** (supporting opposing view): {antithesis_ids}

Please address this conflict in your response by presenting both perspectives.

## User Question
{query}
"""

PASSAGE_TEMPLATE = """[{index}] Source: {source}
{text}
---"""


# --- Prompt Builder ---

class PromptBuilder:
    """Builds prompts for RAG generation."""

    def __init__(self) -> None:
        self.system_prompts = {
            "vanilla": SYSTEM_PROMPT_VANILLA,
            "eb-rag": SYSTEM_PROMPT_EBRAG,
            "benchmark": SYSTEM_PROMPT_BENCHMARK,
        }

    def build_messages(
        self,
        context: RAGContext,
        mode: str = "eb-rag",
        custom_system_prompt: str | None = None,
    ) -> list[Message]:
        """
        Build message list for LLM.

        Args:
            context: RAG context with passages and query
            mode: RAG mode (vanilla, eb-rag, benchmark)
            custom_system_prompt: Optional custom system prompt

        Returns:
            List of messages for LLM
        """
        messages = []

        # System prompt
        system_prompt = custom_system_prompt or self.system_prompts.get(
            mode, SYSTEM_PROMPT_EBRAG
        )
        messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))

        # Add conversation history if present
        for msg in context.conversation_history:
            messages.append(msg)

        # Build context message
        user_content = self._build_context_content(context)
        messages.append(Message(role=MessageRole.USER, content=user_content))

        return messages

    def _build_context_content(self, context: RAGContext) -> str:
        """Build the user message content with context."""
        # Format passages
        passages_text = self._format_passages(context.passages)

        # Choose template based on conflict presence
        if context.has_conflicts and context.conflict_summary:
            return CONTEXT_WITH_CONFLICT_TEMPLATE.format(
                passages=passages_text,
                conflict_summary=context.conflict_summary,
                thesis_ids=", ".join(f"[{i}]" for i in context.thesis_passages) or "N/A",
                antithesis_ids=", ".join(f"[{i}]" for i in context.antithesis_passages) or "N/A",
                query=context.query,
            )
        else:
            return CONTEXT_TEMPLATE.format(
                passages=passages_text,
                query=context.query,
            )

    def _format_passages(self, passages: list[dict[str, Any]]) -> str:
        """Format passages for inclusion in prompt."""
        formatted = []

        for i, passage in enumerate(passages, 1):
            text = passage.get("text", passage.get("content", ""))
            source = passage.get("source", passage.get("source_id", f"Document {i}"))

            formatted.append(PASSAGE_TEMPLATE.format(
                index=i,
                source=source,
                text=text[:1500] + "..." if len(text) > 1500 else text,
            ))

        return "\n".join(formatted)

    def build_synthesis_prompt(
        self,
        query: str,
        thesis_summary: str,
        antithesis_summary: str,
    ) -> str:
        """Build a prompt for synthesizing conflicting information."""
        return f"""Given the following conflicting perspectives on the question:

Question: {query}

Perspective A (Thesis):
{thesis_summary}

Perspective B (Antithesis):
{antithesis_summary}

Please synthesize these perspectives into a balanced response that:
1. Acknowledges both viewpoints
2. Identifies any common ground
3. Explains the source of disagreement if apparent
4. Provides a nuanced conclusion"""

    def build_citation_extraction_prompt(
        self,
        answer: str,
        passages: list[dict[str, Any]],
    ) -> str:
        """Build prompt for extracting citations from answer."""
        passages_text = self._format_passages(passages)

        return f"""Analyze the following answer and identify which passages support each claim.

## Answer:
{answer}

## Available Passages:
{passages_text}

For each factual claim in the answer, identify:
1. The claim text
2. The passage number(s) that support it
3. Whether the passage fully supports, partially supports, or contradicts the claim

Respond in JSON format:
[{{"claim": "...", "passage_ids": [1, 2], "support_level": "full|partial|none"}}]"""

    def build_follow_up_prompt(
        self,
        original_answer: str,
        follow_up_question: str,
        context: RAGContext,
    ) -> list[Message]:
        """Build messages for a follow-up question."""
        messages = [
            Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT_EBRAG),
        ]

        # Add original Q&A as context
        messages.append(Message(
            role=MessageRole.USER,
            content=f"Original question: {context.query}",
        ))
        messages.append(Message(
            role=MessageRole.ASSISTANT,
            content=original_answer,
        ))

        # Add follow-up with fresh context
        follow_up_context = self._build_context_content(context)
        messages.append(Message(
            role=MessageRole.USER,
            content=f"{follow_up_context}\n\nFollow-up question: {follow_up_question}",
        ))

        return messages


# Global prompt builder
_prompt_builder: PromptBuilder | None = None


def get_prompt_builder() -> PromptBuilder:
    """Get the global prompt builder."""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder
