"""Dialectic engine for evidence-balanced reasoning."""

from ebrag.dialectic.models import (
    ConflictType,
    StanceLabel,
    EntailmentLabel,
    ConflictPair,
    ConflictAnalysis,
    Citation,
    ProvenanceRecord,
    ConfidenceScore,
    CalibrationBin,
    CalibrationMetrics,
    SynthesizedContext,
    DialecticResult,
)
from ebrag.dialectic.conflict import (
    ConflictDetector,
    get_conflict_detector,
)
from ebrag.dialectic.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from ebrag.dialectic.calibration import (
    ConfidenceCalibrator,
    get_confidence_calibrator,
)
from ebrag.dialectic.synthesis import (
    SynthesisEngine,
    get_synthesis_engine,
)

__all__ = [
    # Models
    "ConflictType",
    "StanceLabel",
    "EntailmentLabel",
    "ConflictPair",
    "ConflictAnalysis",
    "Citation",
    "ProvenanceRecord",
    "ConfidenceScore",
    "CalibrationBin",
    "CalibrationMetrics",
    "SynthesizedContext",
    "DialecticResult",
    # Conflict detection
    "ConflictDetector",
    "get_conflict_detector",
    # Provenance
    "ProvenanceTracker",
    "get_provenance_tracker",
    # Calibration
    "ConfidenceCalibrator",
    "get_confidence_calibrator",
    # Synthesis
    "SynthesisEngine",
    "get_synthesis_engine",
]
