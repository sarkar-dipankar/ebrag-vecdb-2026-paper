"""
RAG pipeline orchestrator.

Coordinates retrieval, dialectic reasoning, generation, and compliance
into a unified pipeline.
"""

import time
import uuid
from typing import Any, AsyncIterator

from ebrag.common import get_logger, get_settings
from ebrag.chat.llm import BaseLLMClient, LLMProvider, get_llm_client
from ebrag.chat.models import (
    GenerationRequest,
    Message,
    MessageRole,
    RAGContext,
    RAGResponse,
    StreamChunk,
)
from ebrag.chat.prompts import get_prompt_builder

logger = get_logger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline orchestrator.

    Coordinates all components:
    1. Retrieval (dense + sparse + reranking)
    2. Dialectic analysis (conflict detection, provenance)
    3. Generation (with context and conflict awareness)
    4. Compliance (validation, policy, audit)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        llm_provider: LLMProvider = LLMProvider.OPENAI,
        llm_model: str | None = None,
        mode: str = "eb-rag",
    ) -> None:
        self.settings = get_settings()
        self.mode = mode

        # LLM client
        self.llm = llm_client or get_llm_client(llm_provider, llm_model)

        # Prompt builder
        self.prompt_builder = get_prompt_builder()

        # Component flags
        self._retriever_available = False
        self._dialectic_available = False
        self._compliance_available = False

        # Try to load components
        self._init_components()

        logger.info(
            "rag_pipeline_created",
            mode=mode,
            llm_model=self.llm.get_model_name(),
        )

    def _init_components(self) -> None:
        """Initialize available components."""
        # Try to load retriever
        try:
            from ebrag.retrieval import HybridRetriever
            self._retriever_available = True
        except ImportError:
            logger.debug("retrieval_not_available")

        # Try to load dialectic
        try:
            from ebrag.dialectic import SynthesisEngine
            self._dialectic_available = True
        except ImportError:
            logger.debug("dialectic_not_available")

        # Try to load compliance
        try:
            from ebrag.compliance import ComplianceService
            self._compliance_available = True
        except ImportError:
            logger.debug("compliance_not_available")

    async def process(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        conversation_history: list[Message] | None = None,
        enable_dialectic: bool = True,
        enable_compliance: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> RAGResponse:
        """
        Process a query through the full RAG pipeline.

        Args:
            query: User query
            namespace: Retrieval namespace
            top_k: Number of passages to retrieve
            filters: Metadata filters for retrieval
            conversation_history: Previous conversation turns
            enable_dialectic: Enable dialectic reasoning
            enable_compliance: Enable compliance checks
            temperature: Generation temperature
            max_tokens: Max generation tokens

        Returns:
            Complete RAGResponse
        """
        response_id = str(uuid.uuid4())[:12]
        start_time = time.perf_counter()

        logger.info(
            "pipeline_start",
            response_id=response_id,
            query_length=len(query),
            mode=self.mode,
        )

        # Step 1: Retrieval
        passages = await self._retrieve(query, namespace, top_k, filters)
        retrieval_time = time.perf_counter()

        # Step 2: Dialectic Analysis (if enabled and available)
        context = RAGContext(
            query=query,
            passages=passages,
            conversation_history=conversation_history or [],
        )

        if enable_dialectic and self.mode != "vanilla":
            context = await self._analyze_dialectic(context)

        dialectic_time = time.perf_counter()

        # Step 3: Generation
        answer, gen_response = await self._generate(context, temperature, max_tokens)
        generation_time = time.perf_counter()

        # Step 4: Compliance (if enabled and available)
        compliance_result = None
        if enable_compliance:
            compliance_result = await self._check_compliance(
                query=query,
                answer=answer,
                passages=passages,
                response_id=response_id,
            )

        # Apply redaction if needed
        final_answer = answer
        was_redacted = False
        if compliance_result and compliance_result.get("was_redacted"):
            final_answer = compliance_result.get("redacted_text", answer)
            was_redacted = True

        total_time = (time.perf_counter() - start_time) * 1000

        # Build response
        response = RAGResponse(
            response_id=response_id,
            query=query,
            answer=final_answer,
            model=self.llm.get_model_name(),
            generation_time_ms=(generation_time - dialectic_time) * 1000,
            tokens_used=gen_response.total_tokens,
            citations=self._extract_citations(answer, passages),
            confidence=self._calculate_confidence(context, gen_response),
            retrieval_confidence=0.8 if passages else 0.0,  # Placeholder
            generation_confidence=0.8,  # Placeholder
            attribution_confidence=0.7,  # Placeholder
            has_conflicts=context.has_conflicts,
            conflict_count=len(context.thesis_passages) if context.has_conflicts else 0,
            compliance_passed=compliance_result.get("passed", True) if compliance_result else True,
            was_redacted=was_redacted,
        )

        logger.info(
            "pipeline_complete",
            response_id=response_id,
            total_time_ms=total_time,
            retrieval_time_ms=(retrieval_time - start_time) * 1000,
            dialectic_time_ms=(dialectic_time - retrieval_time) * 1000,
            generation_time_ms=(generation_time - dialectic_time) * 1000,
            tokens_used=gen_response.total_tokens,
        )

        return response

    async def _retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant passages."""
        if not self._retriever_available:
            logger.debug("retrieval_skipped_not_available")
            return []

        try:
            import asyncio
            from ebrag.retrieval import get_retriever, RetrievalMode

            # Get retriever for namespace
            retriever = get_retriever(namespace=namespace)

            # Map mode
            mode = RetrievalMode.EBRAG if self.mode != "vanilla" else RetrievalMode.VANILLA

            # Call synchronous retrieve in thread pool
            result = await asyncio.to_thread(
                retriever.retrieve,
                query=query,
                mode=mode,
                k=top_k,
                rerank=True,
            )

            # Map result to expected format
            passages = []
            for p in result.all_passages:
                passages.append({
                    "passage_id": p.id,
                    "source_id": p.source_id,
                    "text": p.text,
                    "score": p.final_score(),
                    "metadata": p.metadata,
                    "stance_score": p.stance_score,
                })

            logger.debug("retrieval_complete", count=len(passages))
            return passages

        except Exception as e:
            logger.error("retrieval_error", error=str(e))
            return []

    async def _analyze_dialectic(self, context: RAGContext) -> RAGContext:
        """Analyze passages for conflicts and prepare synthesis."""
        if not self._dialectic_available or not context.passages:
            return context

        try:
            from ebrag.dialectic import get_synthesis_engine

            engine = get_synthesis_engine()
            synthesis = engine.synthesize(
                query=context.query,
                passages=context.passages,
            )

            context.has_conflicts = synthesis.has_conflicts
            if synthesis.conflicts:
                context.conflict_summary = synthesis.conflicts.summary

                # Categorize passages
                for pair in synthesis.conflicts.conflict_pairs:
                    if pair.thesis_excerpt:
                        context.thesis_passages.append(pair.passage_a_id)
                    if pair.antithesis_excerpt:
                        context.antithesis_passages.append(pair.passage_b_id)

            return context
        except Exception as e:
            logger.error("dialectic_error", error=str(e))
            return context

    async def _generate(
        self,
        context: RAGContext,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, Any]:
        """Generate response using LLM."""
        messages = self.prompt_builder.build_messages(context, self.mode)

        request = GenerationRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response = await self.llm.generate(request)

        return response.content, response

    async def _check_compliance(
        self,
        query: str,
        answer: str,
        passages: list[dict[str, Any]],
        response_id: str,
    ) -> dict[str, Any]:
        """Run compliance checks."""
        if not self._compliance_available:
            return {"passed": True}

        try:
            from ebrag.compliance import get_compliance_service

            service = get_compliance_service()
            result = service.full_compliance_check(
                query=query,
                response_text=answer,
                request_id=response_id,
            )

            return result
        except Exception as e:
            logger.error("compliance_error", error=str(e))
            return {"passed": True}

    def _extract_citations(
        self,
        answer: str,
        passages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract citations from answer."""
        import re

        citations = []
        # Find citation patterns like [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, answer)

        seen = set()
        for match in matches:
            idx = int(match)
            if idx <= len(passages) and idx not in seen:
                seen.add(idx)
                passage = passages[idx - 1]
                citations.append({
                    "citation_id": f"cite-{idx}",
                    "passage_id": passage.get("passage_id", f"p{idx}"),
                    "source_id": passage.get("source_id", ""),
                    "text_excerpt": passage.get("text", "")[:200],
                    "relevance_score": passage.get("score", 0.0),
                })

        return citations

    def _calculate_confidence(
        self,
        context: RAGContext,
        gen_response: Any,
    ) -> float:
        """Calculate overall confidence score."""
        # Base confidence
        confidence = 0.5

        # Adjust based on passage count
        if context.passages:
            confidence += min(0.2, len(context.passages) * 0.02)

        # Reduce for conflicts
        if context.has_conflicts:
            confidence -= 0.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    async def process_stream(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        conversation_history: list[Message] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[StreamChunk]:
        """
        Process a query with streaming response.

        Yields chunks as they are generated.
        """
        response_id = str(uuid.uuid4())[:12]

        logger.info(
            "pipeline_stream_start",
            response_id=response_id,
            query_length=len(query),
        )

        # Retrieve and analyze (blocking)
        passages = await self._retrieve(query, namespace, top_k, filters)

        context = RAGContext(
            query=query,
            passages=passages,
            conversation_history=conversation_history or [],
        )

        if self.mode != "vanilla":
            context = await self._analyze_dialectic(context)

        # Build messages
        messages = self.prompt_builder.build_messages(context, self.mode)

        request = GenerationRequest(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Stream generation
        async for chunk in self.llm.generate_stream(request):
            yield chunk


# Global pipeline instance
_pipeline: RAGPipeline | None = None


def get_rag_pipeline(
    mode: str = "eb-rag",
    llm_provider: LLMProvider = LLMProvider.OPENAI,
    llm_model: str | None = None,
) -> RAGPipeline:
    """Get or create the RAG pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            mode=mode,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
    return _pipeline
