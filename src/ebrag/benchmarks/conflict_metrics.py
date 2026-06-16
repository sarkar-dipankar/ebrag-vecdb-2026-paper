"""
Conflict-aware evaluation metrics used by the C4 experiments.

Pure functions over plain sequences plus small Pydantic result containers. The C4
experiment scripts use only a small slice of these helpers.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
from pydantic import BaseModel, Field

from ebrag.benchmarks.metrics import recall_at_k
from ebrag.common import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Handling behaviour
# --------------------------------------------------------------------------- #


class HandlingBehavior(str, Enum):
    """How a system handled a query, w.r.t. evidence discrepancy."""

    COMMITTED = "committed"  # asserts a single answer, no hedging
    HEDGED = "hedged"  # answers but flags uncertainty
    ABSTAINED = "abstained"  # declines to answer
    PRESENTED_ALTERNATIVES = "presented_alternatives"  # gives multiple valid answers


#: Behaviours that count as *acknowledging* a discrepancy (non-committal).
NON_COMMITTAL_BEHAVIORS: frozenset[HandlingBehavior] = frozenset(
    {
        HandlingBehavior.HEDGED,
        HandlingBehavior.ABSTAINED,
        HandlingBehavior.PRESENTED_ALTERNATIVES,
    }
)


def is_non_committal(behavior: HandlingBehavior | str) -> bool:
    """True if the behaviour acknowledges a discrepancy (hedge/abstain/alternatives)."""
    if isinstance(behavior, str) and not isinstance(behavior, HandlingBehavior):
        behavior = HandlingBehavior(behavior)
    return behavior in NON_COMMITTAL_BEHAVIORS


# --------------------------------------------------------------------------- #
# EDHS: Evidence Discrepancy Handling Score
# --------------------------------------------------------------------------- #


class EDHSResult(BaseModel):
    """Result of an EDHS computation, with the per-stratum components exposed."""

    edhs: float = 0.0
    #: Sensitivity: fraction of *conflicting* queries handled non-committally.
    conflict_handling_rate: float | None = None
    #: Specificity: fraction of *unanimous* queries answered with commitment
    #: (and, in strict mode, correctly).
    commit_rate_unanimous: float | None = None
    n_conflict: int = 0
    n_unanimous: int = 0
    strict: bool = False


def edhs(
    behaviors: Sequence[HandlingBehavior | str],
    conflict_labels: Sequence[bool],
    correct: Sequence[bool] | None = None,
    strict: bool = False,
) -> EDHSResult:
    """Compute the Evidence Discrepancy Handling Score.

    EDHS is the **balanced accuracy of the handling decision** across the two evidence
    regimes, so it cannot be gamed by always-abstaining or always-committing:

    - On *conflicting* queries the system should be **non-committal** (hedge / abstain /
      present alternatives) -> ``conflict_handling_rate`` (sensitivity).
    - On *unanimous* queries the system should **commit** (and, in ``strict`` mode, be
      correct) -> ``commit_rate_unanimous`` (specificity).

    ``edhs = mean(sensitivity, specificity)`` when both strata are non-empty; otherwise
    it falls back to whichever stratum is present. Range ``[0, 1]``; ``0.5`` is the
    trivial single-behaviour baseline.

    Args:
        behaviors: Per-query handling behaviour.
        conflict_labels: Per-query gold label, ``True`` if evidence genuinely conflicts.
        correct: Per-query answer correctness; required when ``strict=True``.
        strict: If set, an unanimous-stratum query only counts as well-handled when it
            is both committed *and* correct.

    Returns:
        :class:`EDHSResult` with the score and its components.
    """
    n = len(behaviors)
    if len(conflict_labels) != n:
        raise ValueError("behaviors and conflict_labels must have the same length")
    if strict and (correct is None or len(correct) != n):
        raise ValueError("strict EDHS requires `correct` with the same length")
    if n == 0:
        return EDHSResult(strict=strict)

    conflict_idx = [i for i in range(n) if conflict_labels[i]]
    unanimous_idx = [i for i in range(n) if not conflict_labels[i]]

    sensitivity: float | None = None
    if conflict_idx:
        handled = sum(1 for i in conflict_idx if is_non_committal(behaviors[i]))
        sensitivity = handled / len(conflict_idx)

    specificity: float | None = None
    if unanimous_idx:
        good = 0
        for i in unanimous_idx:
            committed = not is_non_committal(behaviors[i])
            if strict:
                good += int(committed and bool(correct[i]))  # type: ignore[index]
            else:
                good += int(committed)
        specificity = good / len(unanimous_idx)

    components = [c for c in (sensitivity, specificity) if c is not None]
    score = float(np.mean(components)) if components else 0.0

    result = EDHSResult(
        edhs=score,
        conflict_handling_rate=sensitivity,
        commit_rate_unanimous=specificity,
        n_conflict=len(conflict_idx),
        n_unanimous=len(unanimous_idx),
        strict=strict,
    )
    logger.debug(
        "edhs_computed",
        edhs=round(score, 4),
        sensitivity=sensitivity,
        specificity=specificity,
        n_conflict=len(conflict_idx),
        n_unanimous=len(unanimous_idx),
        strict=strict,
    )
    return result


# --------------------------------------------------------------------------- #
# Counter-evidence / stance recall
# --------------------------------------------------------------------------- #


class StanceRecall(BaseModel):
    """Supporting vs counter-evidence recall and the burial gap."""

    k: int
    supporting_recall: float = 0.0
    counter_recall: float = 0.0
    #: ``supporting_recall - counter_recall``. Positive => counter-evidence is buried.
    recall_gap: float = 0.0


def counter_evidence_recall(
    retrieved_ids: Sequence[str],
    counter_evidence_ids: Sequence[str],
    k: int,
) -> float:
    """Recall@k restricted to gold *counter-evidence* (refuting / minority) passages."""
    return recall_at_k(retrieved_ids, counter_evidence_ids, k)


def stance_recall(
    retrieved_ids: Sequence[str],
    supporting_ids: Sequence[str],
    counter_evidence_ids: Sequence[str],
    k: int,
) -> StanceRecall:
    """Compute supporting vs counter-evidence recall@k and the burial gap (C1 headline).

    A consistently positive ``recall_gap`` across queries is the evidence that standard
    relevance ranking surfaces confirming evidence while burying refuting/minority
    evidence -- the bottleneck C1 targets.
    """
    s = recall_at_k(retrieved_ids, supporting_ids, k)
    c = recall_at_k(retrieved_ids, counter_evidence_ids, k)
    return StanceRecall(k=k, supporting_recall=s, counter_recall=c, recall_gap=s - c)


# --------------------------------------------------------------------------- #
# Calibration primitives (standalone; mirror dialectic/calibration.py logic)
# --------------------------------------------------------------------------- #


def _validate_pairs(confidences: Sequence[float], outcomes: Sequence[bool]) -> None:
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have the same length")


def expected_calibration_error(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    num_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width bins over ``[0, 1]``."""
    _validate_pairs(confidences, outcomes)
    n = len(confidences)
    if n == 0:
        return 0.0

    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so confidence == 1.0 is included
        in_bin = (conf >= lo) & (conf < hi)
        if i == num_bins - 1:
            in_bin |= conf == 1.0
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(conf[in_bin].mean())
        bin_acc = float(acc[in_bin].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(confidences: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Mean squared error between confidence and correctness."""
    _validate_pairs(confidences, outcomes)
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(outcomes, dtype=float)
    return float(np.mean((conf - acc) ** 2))


def risk_coverage_curve(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
) -> tuple[list[float], list[float]]:
    """Selective-prediction risk-coverage curve.

    Answers are released in order of decreasing confidence. At each prefix we report the
    coverage (fraction answered) and the risk (error rate among those answered).

    Returns:
        ``(coverages, risks)`` aligned lists of length ``n``.
    """
    _validate_pairs(confidences, outcomes)
    n = len(confidences)
    if n == 0:
        return [], []

    order = np.argsort(-np.asarray(confidences, dtype=float))
    correct_sorted = np.asarray(outcomes, dtype=float)[order]
    cum_correct = np.cumsum(correct_sorted)
    counts = np.arange(1, n + 1)

    coverages = (counts / n).tolist()
    risks = (1.0 - cum_correct / counts).tolist()
    return coverages, risks


def aurc(confidences: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Area Under the Risk-Coverage curve (lower is better)."""
    coverages, risks = risk_coverage_curve(confidences, outcomes)
    if not coverages:
        return 0.0
    # Trapezoidal integral of risk over coverage, normalised by coverage span.
    cov = np.asarray(coverages, dtype=float)
    rsk = np.asarray(risks, dtype=float)
    area = float(np.trapezoid(rsk, cov)) if hasattr(np, "trapezoid") else float(
        np.trapz(rsk, cov)
    )
    span = cov[-1] - cov[0]
    return area / span if span > 0 else float(rsk.mean())


def coverage_at_risk(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    max_risk: float,
) -> float:
    """Maximum coverage achievable while keeping selective risk <= ``max_risk``."""
    coverages, risks = risk_coverage_curve(confidences, outcomes)
    best = 0.0
    for cov, rsk in zip(coverages, risks):
        if rsk <= max_risk:
            best = max(best, cov)
    return best


# --------------------------------------------------------------------------- #
# Conflict-stratified calibration (C2 headline)
# --------------------------------------------------------------------------- #


class StratifiedCalibration(BaseModel):
    """Calibration reported overall and within the conflict / unanimous strata."""

    overall_ece: float = 0.0
    conflict_ece: float | None = None
    unanimous_ece: float | None = None
    #: ``conflict_ece - unanimous_ece``. Positive => worse-calibrated under conflict.
    ece_gap: float | None = None

    overall_brier: float = 0.0
    conflict_brier: float | None = None
    unanimous_brier: float | None = None

    overall_aurc: float = 0.0
    conflict_aurc: float | None = None
    unanimous_aurc: float | None = None

    n_conflict: int = 0
    n_unanimous: int = 0
    num_bins: int = 10


def stratified_calibration(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    conflict_labels: Sequence[bool],
    num_bins: int = 10,
) -> StratifiedCalibration:
    """Compute calibration overall and per evidence regime (C2 headline).

    The ``ece_gap`` (conflict ECE minus unanimous ECE) is the quantity C2 argues the
    marginal ``overall_ece`` hides: models can look calibrated on aggregate yet be badly
    overconfident on the subset where evidence conflicts.
    """
    n = len(confidences)
    if len(outcomes) != n or len(conflict_labels) != n:
        raise ValueError("confidences, outcomes, conflict_labels must share length")

    conf = list(confidences)
    out = list(outcomes)

    def _subset(mask_true: bool) -> tuple[list[float], list[bool]]:
        c = [conf[i] for i in range(n) if bool(conflict_labels[i]) is mask_true]
        o = [out[i] for i in range(n) if bool(conflict_labels[i]) is mask_true]
        return c, o

    c_conf, c_out = _subset(True)
    u_conf, u_out = _subset(False)

    conflict_ece = (
        expected_calibration_error(c_conf, c_out, num_bins) if c_conf else None
    )
    unanimous_ece = (
        expected_calibration_error(u_conf, u_out, num_bins) if u_conf else None
    )
    ece_gap = (
        conflict_ece - unanimous_ece
        if (conflict_ece is not None and unanimous_ece is not None)
        else None
    )

    return StratifiedCalibration(
        overall_ece=expected_calibration_error(conf, out, num_bins),
        conflict_ece=conflict_ece,
        unanimous_ece=unanimous_ece,
        ece_gap=ece_gap,
        overall_brier=brier_score(conf, out),
        conflict_brier=brier_score(c_conf, c_out) if c_conf else None,
        unanimous_brier=brier_score(u_conf, u_out) if u_conf else None,
        overall_aurc=aurc(conf, out),
        conflict_aurc=aurc(c_conf, c_out) if c_conf else None,
        unanimous_aurc=aurc(u_conf, u_out) if u_conf else None,
        n_conflict=len(c_conf),
        n_unanimous=len(u_conf),
        num_bins=num_bins,
    )
