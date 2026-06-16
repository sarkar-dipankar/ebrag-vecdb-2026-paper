"""
Benchmark models and data structures.

Defines the core types for benchmark evaluation including
questions, results, and aggregated metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkType(str, Enum):
    """Types of benchmarks supported."""

    MULTI_HOP = "multi_hop"  # HotpotQA
    REASONING = "reasoning"  # StrategyQA, GSM8K
    FACTUAL = "factual"  # TruthfulQA, FEVER
    SCIENTIFIC = "scientific"  # PubMedQA, SciFact
    LONG_FORM = "long_form"  # Qasper
    TOOL_CALLING = "tool_calling"  # ToolBench, GAIA


class AnswerType(str, Enum):
    """Types of expected answers."""

    EXTRACTIVE = "extractive"  # Answer is a span from context
    ABSTRACTIVE = "abstractive"  # Answer is generated
    BOOLEAN = "boolean"  # Yes/No answer
    CLASSIFICATION = "classification"  # Category label
    NUMERIC = "numeric"  # Number or calculation


class BenchmarkQuestion(BaseModel):
    """A single benchmark question."""

    id: str
    question: str
    answer: str | list[str]  # Gold answer(s)
    answer_type: AnswerType = AnswerType.EXTRACTIVE

    # Optional context for closed-book evaluation
    context: str | None = None
    supporting_facts: list[str] = Field(default_factory=list)

    # Metadata
    dataset_name: str = ""
    difficulty: str | None = None
    topic: str | None = None

    # For multi-hop questions
    sub_questions: list[str] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval quality."""

    # Recall metrics
    recall_at_k: dict[int, float] = Field(default_factory=dict)  # {k: recall}

    # Precision metrics
    precision_at_k: dict[int, float] = Field(default_factory=dict)

    # Mean Reciprocal Rank
    mrr: float = 0.0

    # Normalized Discounted Cumulative Gain
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)

    # Answer presence in retrieved context
    answer_in_context: bool = False
    answer_position: int | None = None  # Position in retrieved passages

    # EB-RAG specific metrics
    thesis_recall: float = 0.0
    antithesis_recall: float = 0.0
    diversity_score: float = 0.0
    conflict_detected: bool = False


class AnswerMetrics(BaseModel):
    """Metrics for answer quality."""

    # Exact match
    exact_match: float = 0.0

    # Token-level F1
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # ROUGE scores
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0

    # For boolean/classification
    accuracy: float = 0.0

    # Semantic similarity (embedding-based)
    semantic_similarity: float = 0.0

    # Faithfulness (answer grounded in retrieved context)
    faithfulness: float = 0.0


class QuestionResult(BaseModel):
    """Result for a single question."""

    question_id: str
    question: str
    gold_answer: str | list[str]
    predicted_answer: str

    # Metrics
    retrieval_metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    answer_metrics: AnswerMetrics = Field(default_factory=AnswerMetrics)

    # Retrieved passages
    retrieved_passages: list[str] = Field(default_factory=list)
    thesis_passages: list[str] = Field(default_factory=list)
    antithesis_passages: list[str] = Field(default_factory=list)

    # Timing
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Error tracking
    error: str | None = None

    # LLM response details
    llm_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    # Dataset settings
    dataset_name: str
    split: str = "test"
    max_samples: int | None = None  # None = all samples

    # Retrieval settings
    retrieval_mode: str = "eb-rag"  # "vanilla" or "eb-rag"
    top_k: int = 5
    use_reranking: bool = True

    # Generation settings
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512

    # Evaluation settings
    metrics: list[str] = Field(
        default_factory=lambda: ["exact_match", "f1", "rouge_l"]
    )

    # Comparison settings
    compare_modes: bool = True  # Compare vanilla vs eb-rag

    # Random seed for reproducibility
    seed: int = 42


class BenchmarkResult(BaseModel):
    """Aggregated results for a benchmark run."""

    # Run metadata
    run_id: str
    benchmark_name: str
    dataset_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Configuration used
    config: BenchmarkConfig

    # Per-question results
    question_results: list[QuestionResult] = Field(default_factory=list)

    # Aggregated retrieval metrics
    avg_retrieval_metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)

    # Aggregated answer metrics
    avg_answer_metrics: AnswerMetrics = Field(default_factory=AnswerMetrics)

    # Summary statistics
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0

    # Timing
    avg_retrieval_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    total_run_time_s: float = 0.0

    # Token usage
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # EB-RAG specific
    avg_diversity_score: float = 0.0
    conflict_rate: float = 0.0  # % of questions with detected conflicts


class ComparisonResult(BaseModel):
    """Comparison between vanilla and EB-RAG modes."""

    # Run metadata
    run_id: str
    dataset_name: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Results for each mode
    vanilla_result: BenchmarkResult | None = None
    ebrag_result: BenchmarkResult | None = None

    # Metric deltas (eb-rag - vanilla)
    exact_match_delta: float = 0.0
    f1_delta: float = 0.0
    rouge_l_delta: float = 0.0
    retrieval_recall_delta: float = 0.0

    # Statistical significance (p-values)
    em_p_value: float | None = None
    f1_p_value: float | None = None

    # EB-RAG advantage breakdown
    improvement_on_controversial: float = 0.0  # Questions with conflicting evidence
    improvement_on_multi_perspective: float = 0.0  # Questions needing multiple views


@dataclass
class DatasetInfo:
    """Information about a benchmark dataset."""

    name: str
    benchmark_type: BenchmarkType
    answer_type: AnswerType
    description: str

    # Dataset size
    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0

    # Source information
    source_url: str = ""
    citation: str = ""

    # Dataset characteristics
    avg_question_length: float = 0.0
    avg_answer_length: float = 0.0
    requires_reasoning: bool = False
    multi_hop: bool = False

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


# Dataset registry with metadata
DATASET_INFO: dict[str, DatasetInfo] = {
    "hotpotqa": DatasetInfo(
        name="HotpotQA",
        benchmark_type=BenchmarkType.MULTI_HOP,
        answer_type=AnswerType.EXTRACTIVE,
        description="Multi-hop question answering requiring reasoning over multiple documents",
        train_size=90447,
        validation_size=7405,
        test_size=7405,
        source_url="https://hotpotqa.github.io/",
        multi_hop=True,
        requires_reasoning=True,
    ),
    "strategyqa": DatasetInfo(
        name="StrategyQA",
        benchmark_type=BenchmarkType.REASONING,
        answer_type=AnswerType.BOOLEAN,
        description="Yes/No questions requiring implicit multi-step reasoning",
        train_size=2290,
        test_size=490,
        source_url="https://allenai.org/data/strategyqa",
        requires_reasoning=True,
    ),
    "truthfulqa": DatasetInfo(
        name="TruthfulQA",
        benchmark_type=BenchmarkType.FACTUAL,
        answer_type=AnswerType.ABSTRACTIVE,
        description="Questions designed to test truthfulness against common misconceptions",
        test_size=817,
        source_url="https://github.com/sylinrl/TruthfulQA",
    ),
    "fever": DatasetInfo(
        name="FEVER",
        benchmark_type=BenchmarkType.FACTUAL,
        answer_type=AnswerType.CLASSIFICATION,
        description="Fact verification requiring evidence retrieval",
        train_size=145449,
        validation_size=19998,
        test_size=19998,
        source_url="https://fever.ai/",
    ),
    "pubmedqa": DatasetInfo(
        name="PubMedQA",
        benchmark_type=BenchmarkType.SCIENTIFIC,
        answer_type=AnswerType.BOOLEAN,
        description="Biomedical yes/no questions based on PubMed abstracts",
        train_size=450,
        validation_size=50,
        test_size=500,
        source_url="https://pubmedqa.github.io/",
        requires_reasoning=True,
    ),
    "scifact": DatasetInfo(
        name="SciFact",
        benchmark_type=BenchmarkType.SCIENTIFIC,
        answer_type=AnswerType.CLASSIFICATION,
        description="Scientific claim verification with evidence",
        train_size=809,
        validation_size=339,
        test_size=300,
        source_url="https://github.com/allenai/scifact",
    ),
    "qasper": DatasetInfo(
        name="Qasper",
        benchmark_type=BenchmarkType.LONG_FORM,
        answer_type=AnswerType.ABSTRACTIVE,
        description="Question answering over NLP research papers",
        train_size=1585,
        validation_size=239,
        test_size=240,
        source_url="https://allenai.org/data/qasper",
        requires_reasoning=True,
    ),
    "gsm8k": DatasetInfo(
        name="GSM8K",
        benchmark_type=BenchmarkType.REASONING,
        answer_type=AnswerType.NUMERIC,
        description="Grade school math word problems",
        train_size=7473,
        test_size=1319,
        source_url="https://github.com/openai/grade-school-math",
        requires_reasoning=True,
    ),
}
