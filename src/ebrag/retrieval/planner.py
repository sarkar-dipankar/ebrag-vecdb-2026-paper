"""
Query planning for evidence-balanced retrieval.

Generates thesis and antithesis queries to encourage diverse
evidence retrieval that covers multiple perspectives.
"""

import re
from typing import Any

from ebrag.common import get_logger, get_settings
from ebrag.retrieval.models import QueryIntent, QueryPlan, RetrievalMode

logger = get_logger(__name__)


class QueryPlanner:
    """
    Plans retrieval queries for evidence-balanced search.

    Generates thesis (affirmative) and antithesis (skeptical) query
    reformulations to maximize evidence diversity and expose potential
    conflicts in the source material.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        # Keywords that suggest different intents
        self._factual_keywords = {
            "what", "who", "when", "where", "which", "how many", "how much",
            "define", "describe", "explain", "list", "name",
        }
        self._comparative_keywords = {
            "compare", "contrast", "difference", "similar", "versus", "vs",
            "better", "worse", "more", "less", "advantage", "disadvantage",
        }
        self._causal_keywords = {
            "why", "cause", "effect", "result", "because", "reason",
            "lead to", "due to", "consequence", "impact",
        }
        self._opinion_keywords = {
            "should", "would", "could", "opinion", "think", "believe",
            "best", "worst", "recommend", "prefer",
        }

        # Antithesis transformation patterns
        self._negation_pairs = [
            ("is", "is not"),
            ("are", "are not"),
            ("was", "was not"),
            ("were", "were not"),
            ("does", "does not"),
            ("do", "do not"),
            ("can", "cannot"),
            ("will", "will not"),
            ("should", "should not"),
            ("always", "never"),
            ("every", "no"),
            ("all", "none"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("right", "wrong"),
            ("good", "bad"),
            ("benefit", "harm"),
            ("advantage", "disadvantage"),
            ("success", "failure"),
            ("safe", "dangerous"),
            ("effective", "ineffective"),
            ("support", "oppose"),
            ("agree", "disagree"),
            ("confirm", "deny"),
            ("prove", "disprove"),
        ]

    def plan(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.EBRAG,
    ) -> QueryPlan:
        """
        Create a query plan for retrieval.

        Args:
            query: The user's original query
            mode: Retrieval mode (vanilla uses single query)

        Returns:
            QueryPlan with thesis and optional antithesis queries
        """
        # Analyze the query
        intent = self._classify_intent(query)
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query)

        # Generate thesis query (the original or slightly enhanced)
        thesis_query = self._generate_thesis_query(query, intent)

        # Generate antithesis query for EB-RAG mode
        antithesis_query = None
        if mode in (RetrievalMode.EBRAG, RetrievalMode.BENCHMARK):
            antithesis_query = self._generate_antithesis_query(query, intent)

        plan = QueryPlan(
            original_query=query,
            thesis_query=thesis_query,
            antithesis_query=antithesis_query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            thesis_k=self.settings.retrieval.thesis_k,
            antithesis_k=self.settings.retrieval.antithesis_k,
            use_reranking=self.settings.retrieval.use_cross_encoder,
        )

        logger.debug(
            "query_plan_created",
            original=query,
            thesis=thesis_query,
            antithesis=antithesis_query,
            intent=intent.value,
        )

        return plan

    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a query."""
        query_lower = query.lower()

        # Check for comparative queries
        for keyword in self._comparative_keywords:
            if keyword in query_lower:
                return QueryIntent.COMPARATIVE

        # Check for causal queries
        for keyword in self._causal_keywords:
            if keyword in query_lower:
                return QueryIntent.CAUSAL

        # Check for opinion queries
        for keyword in self._opinion_keywords:
            if keyword in query_lower:
                return QueryIntent.OPINION

        # Check for factual queries
        for keyword in self._factual_keywords:
            if query_lower.startswith(keyword) or f" {keyword} " in query_lower:
                return QueryIntent.FACTUAL

        return QueryIntent.UNKNOWN

    def _extract_entities(self, query: str) -> list[str]:
        """Extract potential named entities from query."""
        # Simple heuristic: capitalized words that aren't at start of sentence
        entities = []

        words = query.split()
        for i, word in enumerate(words):
            # Skip first word and common words
            if i == 0:
                continue

            # Check if capitalized (potential entity)
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper():
                entities.append(clean_word)

        # Also extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        return entities

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from query."""
        # Simple keyword extraction based on word frequency patterns
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "it", "its", "this", "that", "these", "those",
            "i", "you", "he", "she", "we", "they", "what", "which", "who",
            "when", "where", "why", "how", "and", "or", "but", "in", "on",
            "at", "to", "for", "of", "with", "by", "from", "as", "about",
        }

        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        return keywords[:10]  # Limit to top 10

    def _generate_thesis_query(self, query: str, intent: QueryIntent) -> str:
        """Generate the thesis (affirmative) query."""
        # For most intents, the original query works well
        # We can enhance it slightly based on intent

        if intent == QueryIntent.COMPARATIVE:
            # Add "differences and similarities" framing
            return f"{query} (similarities and differences)"

        if intent == QueryIntent.CAUSAL:
            # Emphasize causal relationships
            return f"{query} (causes and effects)"

        if intent == QueryIntent.OPINION:
            # Add "arguments for" framing
            if "should" in query.lower():
                return f"{query} (reasons and benefits)"

        # Default: use original query
        return query

    def _generate_antithesis_query(self, query: str, intent: QueryIntent) -> str:
        """Generate the antithesis (skeptical) query."""
        query_lower = query.lower()

        # Strategy 1: Negate key terms
        antithesis = query
        for positive, negative in self._negation_pairs:
            # Check for whole word match
            pattern = rf'\b{positive}\b'
            if re.search(pattern, query_lower):
                antithesis = re.sub(pattern, negative, antithesis, flags=re.IGNORECASE)
                break

        # If no negation applied, use alternative strategies
        if antithesis == query:
            antithesis = self._apply_skeptical_framing(query, intent)

        return antithesis

    def _apply_skeptical_framing(self, query: str, intent: QueryIntent) -> str:
        """Apply skeptical framing to a query."""

        if intent == QueryIntent.FACTUAL:
            # Add contradiction/dispute framing
            return f"criticism of {query}" if len(query) < 50 else f"{query} (disputed claims)"

        if intent == QueryIntent.COMPARATIVE:
            # Flip the comparison perspective
            return f"{query} (counterarguments)"

        if intent == QueryIntent.CAUSAL:
            # Question the causation
            return f"{query} (alternative explanations)"

        if intent == QueryIntent.OPINION:
            # Seek opposing views
            return f"against {query}" if len(query) < 50 else f"{query} (opposing views)"

        # Default: add skeptical keywords
        skeptical_prefixes = [
            "problems with",
            "criticism of",
            "limitations of",
            "alternatives to",
            "counterarguments to",
        ]

        # Choose appropriate prefix based on query structure
        if query.lower().startswith(("what", "who", "when", "where")):
            return f"{query} (controversies and disputes)"
        elif query.lower().startswith(("how", "why")):
            return f"{query} (alternative perspectives)"
        else:
            return f"problems with {query}"

    def expand_query(self, query: str, synonyms: bool = True) -> list[str]:
        """
        Expand a query into multiple related queries.

        Useful for improving recall in retrieval.
        """
        expansions = [query]

        if synonyms:
            # Add simple reformulations
            if "?" in query:
                # Convert question to statement
                statement = query.rstrip("?").strip()
                expansions.append(statement)

            # Add keyword-focused version
            keywords = self._extract_keywords(query)
            if len(keywords) >= 2:
                expansions.append(" ".join(keywords))

        return expansions


# Global planner instance
_planner: QueryPlanner | None = None


def get_query_planner() -> QueryPlanner:
    """Get the global query planner."""
    global _planner
    if _planner is None:
        _planner = QueryPlanner()
    return _planner
