"""
Confidence calibration for prediction reliability.

Implements Expected Calibration Error (ECE) and
temperature scaling for confidence calibration.
"""

from typing import Any

import numpy as np

from ebrag.common import get_logger, get_settings
from ebrag.dialectic.models import (
    CalibrationBin,
    CalibrationMetrics,
    ConfidenceScore,
)

logger = get_logger(__name__)


class ConfidenceCalibrator:
    """
    Calibrates confidence scores for better reliability.

    Uses temperature scaling and binning to improve
    the alignment between confidence and accuracy.
    """

    def __init__(
        self,
        num_bins: int = 10,
        temperature: float = 1.0,
    ) -> None:
        self.settings = get_settings()
        self.num_bins = num_bins
        self.temperature = temperature

        # Learned calibration parameters
        self._calibration_map: dict[int, float] = {}
        self._is_fitted = False

        logger.info(
            "confidence_calibrator_created",
            num_bins=num_bins,
            temperature=temperature,
        )

    def calibrate(
        self,
        raw_confidence: float,
        retrieval_score: float = 0.0,
        generation_score: float = 0.0,
        attribution_score: float = 0.0,
    ) -> ConfidenceScore:
        """
        Calibrate a raw confidence score.

        Args:
            raw_confidence: Original model confidence (0-1)
            retrieval_score: Score from retrieval stage
            generation_score: Score from generation stage
            attribution_score: Score from attribution/citation

        Returns:
            Calibrated ConfidenceScore
        """
        # Apply temperature scaling
        scaled = self._apply_temperature(raw_confidence)

        # Apply learned calibration if fitted
        if self._is_fitted:
            bin_idx = self._get_bin_index(scaled)
            calibrated = self._calibration_map.get(bin_idx, scaled)
        else:
            calibrated = scaled
            bin_idx = self._get_bin_index(scaled)

        # Combine component scores
        component_confidence = self._combine_components(
            retrieval_score,
            generation_score,
            attribution_score,
        )

        # Blend with component confidence if available
        if component_confidence > 0:
            calibrated = 0.7 * calibrated + 0.3 * component_confidence

        # Determine uncertainty
        is_uncertain = calibrated < 0.5 or self._detect_uncertainty(
            raw_confidence,
            retrieval_score,
            attribution_score,
        )

        uncertainty_reason = ""
        if is_uncertain:
            uncertainty_reason = self._get_uncertainty_reason(
                calibrated,
                retrieval_score,
                attribution_score,
            )

        return ConfidenceScore(
            raw_score=raw_confidence,
            calibrated_score=calibrated,
            bin_index=bin_idx,
            retrieval_confidence=retrieval_score,
            generation_confidence=generation_score,
            attribution_confidence=attribution_score,
            is_uncertain=is_uncertain,
            uncertainty_reason=uncertainty_reason,
        )

    def fit(
        self,
        confidences: list[float],
        outcomes: list[bool],
    ) -> CalibrationMetrics:
        """
        Fit the calibrator on historical data.

        Args:
            confidences: List of confidence scores
            outcomes: List of correct/incorrect outcomes

        Returns:
            CalibrationMetrics after fitting
        """
        if len(confidences) != len(outcomes):
            raise ValueError("Confidences and outcomes must have same length")

        if not confidences:
            logger.warning("empty_calibration_data")
            return CalibrationMetrics()

        # Create bins
        bins: list[CalibrationBin] = []
        bin_edges = np.linspace(0, 1, self.num_bins + 1)

        for i in range(self.num_bins):
            bin_lower = float(bin_edges[i])
            bin_upper = float(bin_edges[i + 1])

            # Get samples in this bin
            in_bin = [
                (c, o) for c, o in zip(confidences, outcomes)
                if bin_lower <= c < bin_upper or (i == self.num_bins - 1 and c == 1.0)
            ]

            if in_bin:
                bin_confidences = [c for c, _ in in_bin]
                bin_outcomes = [o for _, o in in_bin]

                bin_accuracy = sum(bin_outcomes) / len(bin_outcomes)
                bin_confidence = sum(bin_confidences) / len(bin_confidences)

                # Store calibration map: bin_idx -> observed accuracy
                self._calibration_map[i] = bin_accuracy
            else:
                bin_accuracy = 0.0
                bin_confidence = (bin_lower + bin_upper) / 2
                self._calibration_map[i] = bin_confidence

            bins.append(CalibrationBin(
                bin_lower=bin_lower,
                bin_upper=bin_upper,
                bin_count=len(in_bin),
                bin_accuracy=bin_accuracy,
                bin_confidence=bin_confidence,
            ))

        self._is_fitted = True

        # Calculate ECE
        ece = self._calculate_ece(bins, len(confidences))
        max_ce = max(
            abs(b.bin_accuracy - b.bin_confidence)
            for b in bins if b.bin_count > 0
        ) if any(b.bin_count > 0 for b in bins) else 0.0

        # Calculate Brier score
        brier = sum(
            (c - (1.0 if o else 0.0)) ** 2
            for c, o in zip(confidences, outcomes)
        ) / len(confidences)

        # Build reliability diagram data
        reliability_data = [
            (b.bin_confidence, b.bin_accuracy)
            for b in bins if b.bin_count > 0
        ]

        metrics = CalibrationMetrics(
            ece=ece,
            max_calibration_error=max_ce,
            bins=bins,
            brier_score=brier,
            reliability_diagram_data=reliability_data,
        )

        logger.info(
            "calibrator_fitted",
            samples=len(confidences),
            ece=round(ece, 4),
            brier=round(brier, 4),
        )

        return metrics

    def _apply_temperature(self, confidence: float) -> float:
        """Apply temperature scaling to confidence."""
        if self.temperature == 1.0:
            return confidence

        # Avoid division by zero
        if confidence <= 0 or confidence >= 1:
            return confidence

        # Convert to logit, scale, convert back
        logit = np.log(confidence / (1 - confidence))
        scaled_logit = logit / self.temperature
        scaled = 1 / (1 + np.exp(-scaled_logit))

        return float(np.clip(scaled, 0, 1))

    def _get_bin_index(self, confidence: float) -> int:
        """Get the bin index for a confidence value."""
        if confidence >= 1.0:
            return self.num_bins - 1
        if confidence <= 0.0:
            return 0

        return int(confidence * self.num_bins)

    def _combine_components(
        self,
        retrieval: float,
        generation: float,
        attribution: float,
    ) -> float:
        """Combine component confidence scores."""
        scores = [s for s in [retrieval, generation, attribution] if s > 0]
        if not scores:
            return 0.0

        # Weighted geometric mean
        return float(np.prod(scores) ** (1 / len(scores)))

    def _detect_uncertainty(
        self,
        raw_confidence: float,
        retrieval_score: float,
        attribution_score: float,
    ) -> bool:
        """Detect if the prediction should be marked uncertain."""
        # Low retrieval quality
        if retrieval_score > 0 and retrieval_score < 0.3:
            return True

        # Low attribution (not grounded in sources)
        if attribution_score > 0 and attribution_score < 0.4:
            return True

        # High raw confidence but low supporting evidence
        if raw_confidence > 0.8 and retrieval_score < 0.5:
            return True

        return False

    def _get_uncertainty_reason(
        self,
        calibrated: float,
        retrieval_score: float,
        attribution_score: float,
    ) -> str:
        """Get human-readable uncertainty reason."""
        reasons = []

        if calibrated < 0.3:
            reasons.append("low overall confidence")
        if retrieval_score > 0 and retrieval_score < 0.3:
            reasons.append("weak retrieval evidence")
        if attribution_score > 0 and attribution_score < 0.4:
            reasons.append("poor source attribution")

        return "; ".join(reasons) if reasons else "uncertain"

    def _calculate_ece(
        self,
        bins: list[CalibrationBin],
        total_samples: int,
    ) -> float:
        """Calculate Expected Calibration Error."""
        if total_samples == 0:
            return 0.0

        ece = 0.0
        for bin_data in bins:
            if bin_data.bin_count > 0:
                weight = bin_data.bin_count / total_samples
                gap = abs(bin_data.bin_accuracy - bin_data.bin_confidence)
                ece += weight * gap

        return float(ece)

    def optimize_temperature(
        self,
        confidences: list[float],
        outcomes: list[bool],
        temps: list[float] | None = None,
    ) -> float:
        """
        Find optimal temperature for calibration.

        Args:
            confidences: Validation confidence scores
            outcomes: Validation outcomes
            temps: Temperature values to try

        Returns:
            Optimal temperature value
        """
        temps = temps or [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

        best_temp = 1.0
        best_ece = float("inf")

        for temp in temps:
            self.temperature = temp
            scaled = [self._apply_temperature(c) for c in confidences]

            # Calculate ECE for this temperature
            temp_bins: list[CalibrationBin] = []
            bin_edges = np.linspace(0, 1, self.num_bins + 1)

            for i in range(self.num_bins):
                bin_lower = float(bin_edges[i])
                bin_upper = float(bin_edges[i + 1])

                in_bin = [
                    (c, o) for c, o in zip(scaled, outcomes)
                    if bin_lower <= c < bin_upper or (i == self.num_bins - 1 and c == 1.0)
                ]

                if in_bin:
                    bin_acc = sum(o for _, o in in_bin) / len(in_bin)
                    bin_conf = sum(c for c, _ in in_bin) / len(in_bin)
                else:
                    bin_acc = 0.0
                    bin_conf = (bin_lower + bin_upper) / 2

                temp_bins.append(CalibrationBin(
                    bin_lower=bin_lower,
                    bin_upper=bin_upper,
                    bin_count=len(in_bin),
                    bin_accuracy=bin_acc,
                    bin_confidence=bin_conf,
                ))

            ece = self._calculate_ece(temp_bins, len(confidences))

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.temperature = best_temp
        logger.info("temperature_optimized", temperature=best_temp, ece=round(best_ece, 4))

        return best_temp


# Global calibrator instance
_calibrator: ConfidenceCalibrator | None = None


def get_confidence_calibrator() -> ConfidenceCalibrator:
    """Get the global confidence calibrator."""
    global _calibrator
    if _calibrator is None:
        _calibrator = ConfidenceCalibrator()
    return _calibrator
