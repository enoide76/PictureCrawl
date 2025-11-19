"""
Market valuation engine for artworks.

Estimates market value based on artist, style, condition, and comparable sales.
"""
import json
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np

from backend.core.config import settings
from backend.core.logging import log
from backend.core.models import ArtistCandidate, ConditionAnalysis, ValuationResult


class ValuationEngine:
    """
    Estimates market value of artworks using historical price data and heuristics.
    """

    def __init__(self, historical_data_path: Optional[Path] = None):
        """
        Initialize valuation engine.

        Args:
            historical_data_path: Path to historical price data JSON
        """
        self.historical_data_path = historical_data_path or (
            settings.HISTORICAL_PRICES_DIR / "sample_prices.json"
        )

        self.historical_prices = []
        self._load_historical_data()

    def _load_historical_data(self):
        """Load historical price data from JSON file."""
        if not self.historical_data_path.exists():
            log.warning(f"Historical price data not found at {self.historical_data_path}")
            return

        with open(self.historical_data_path, "r", encoding="utf-8") as f:
            self.historical_prices = json.load(f)

        log.info(f"Loaded {len(self.historical_prices)} historical price records")

    def estimate_value(
        self,
        artist_candidates: List[ArtistCandidate],
        style: str,
        epoch: str,
        condition: ConditionAnalysis,
        authenticity_score: int,
        size_category: str = "medium"
    ) -> ValuationResult:
        """
        Estimate market value of an artwork.

        Args:
            artist_candidates: List of potential artists
            style: Art style
            epoch: Art epoch
            condition: Condition analysis
            authenticity_score: Authenticity score (0-100)
            size_category: Size category (small, medium, large)

        Returns:
            ValuationResult with estimated value and confidence
        """
        log.info("Estimating artwork value...")

        if not artist_candidates:
            return self._estimate_unknown_artist(style, epoch, condition)

        # Get comparable sales
        comparables = self._find_comparable_sales(
            artist_candidates, style, epoch, size_category
        )

        if not comparables:
            return self._estimate_without_comparables(
                artist_candidates, style, epoch, condition, authenticity_score
            )

        # Calculate base value from comparables
        base_value = np.mean([c["price_realized_eur"] for c in comparables])

        # Apply adjustments
        adjusted_value = self._apply_adjustments(
            base_value,
            condition,
            authenticity_score,
            artist_candidates[0].similarity
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            len(comparables),
            artist_candidates[0].similarity,
            authenticity_score
        )

        # Calculate range
        min_value, max_value = self._calculate_range(adjusted_value, confidence)

        # Generate rationale
        rationale = self._generate_rationale(
            artist_candidates[0],
            len(comparables),
            condition,
            authenticity_score
        )

        result = ValuationResult(
            estimated_value=float(adjusted_value),
            min=float(min_value),
            max=float(max_value),
            confidence=confidence,
            rationale=rationale,
            comparable_sales=len(comparables)
        )

        log.info(f"Estimated value: €{adjusted_value:,.0f} (confidence: {confidence})")
        return result

    def _find_comparable_sales(
        self,
        artist_candidates: List[ArtistCandidate],
        style: str,
        epoch: str,
        size_category: str
    ) -> List[dict]:
        """Find comparable sales from historical data."""
        comparables = []

        # Get artist names
        artist_names = [a.name for a in artist_candidates[:3]]

        for record in self.historical_prices:
            # Match by artist
            if record["artist"] in artist_names:
                comparables.append(record)
                continue

            # Match by style/epoch and size
            if (record.get("style") == style or record.get("epoch") == epoch):
                if record.get("size_category") == size_category:
                    comparables.append(record)

        log.debug(f"Found {len(comparables)} comparable sales")
        return comparables

    def _apply_adjustments(
        self,
        base_value: float,
        condition: ConditionAnalysis,
        authenticity_score: int,
        artist_similarity: float
    ) -> float:
        """Apply adjustments to base value based on various factors."""
        adjusted_value = base_value

        # Condition adjustment
        # damage_score: 0 = mint, 1 = heavily damaged
        condition_factor = 1.0 - (condition.damage_score * 0.5)
        adjusted_value *= condition_factor

        # Authenticity adjustment
        # 100 = certain, 0 = fake
        authenticity_factor = 0.5 + (authenticity_score / 200)  # Range: 0.5 to 1.0
        adjusted_value *= authenticity_factor

        # Artist similarity adjustment
        similarity_factor = 0.7 + (artist_similarity * 0.3)  # Range: 0.7 to 1.0
        adjusted_value *= similarity_factor

        log.debug(
            f"Adjustments: condition={condition_factor:.2f}, "
            f"authenticity={authenticity_factor:.2f}, "
            f"similarity={similarity_factor:.2f}"
        )

        return max(0, adjusted_value)

    def _calculate_confidence(
        self,
        num_comparables: int,
        artist_similarity: float,
        authenticity_score: int
    ) -> Literal["low", "medium", "high"]:
        """Calculate confidence level for valuation."""
        # Score based on multiple factors
        score = 0

        # Number of comparable sales
        if num_comparables >= 10:
            score += 3
        elif num_comparables >= 5:
            score += 2
        elif num_comparables >= 1:
            score += 1

        # Artist match quality
        if artist_similarity > 0.8:
            score += 2
        elif artist_similarity > 0.6:
            score += 1

        # Authenticity confidence
        if authenticity_score > 80:
            score += 2
        elif authenticity_score > 60:
            score += 1

        # Determine confidence level
        if score >= 6:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"

    def _calculate_range(
        self,
        estimated_value: float,
        confidence: Literal["low", "medium", "high"]
    ) -> tuple[float, float]:
        """Calculate min/max value range based on confidence."""
        if confidence == "high":
            range_factor = 0.15  # ±15%
        elif confidence == "medium":
            range_factor = 0.30  # ±30%
        else:
            range_factor = 0.50  # ±50%

        min_value = estimated_value * (1 - range_factor)
        max_value = estimated_value * (1 + range_factor)

        return min_value, max_value

    def _generate_rationale(
        self,
        top_artist: ArtistCandidate,
        num_comparables: int,
        condition: ConditionAnalysis,
        authenticity_score: int
    ) -> str:
        """Generate human-readable rationale for valuation."""
        parts = []

        # Comparable sales
        if num_comparables > 0:
            parts.append(
                f"Based on {num_comparables} similar sale{'s' if num_comparables > 1 else ''}"
            )
        else:
            parts.append("Estimated based on style and epoch without direct comparables")

        # Artist match
        if top_artist.similarity > 0.8:
            parts.append(f"High similarity to {top_artist.name} works")
        elif top_artist.similarity > 0.6:
            parts.append(f"Moderate similarity to {top_artist.name} style")

        # Condition
        if condition.damage_score < 0.2:
            parts.append("Good condition")
        elif condition.damage_score < 0.5:
            parts.append("Fair condition with some aging")
        else:
            parts.append("Condition issues reduce value")

        # Authenticity
        if authenticity_score >= 80:
            parts.append("High authenticity confidence")
        elif authenticity_score >= 60:
            parts.append("Moderate authenticity confidence")
        else:
            parts.append("Low authenticity confidence significantly impacts value")

        return ". ".join(parts) + "."

    def _estimate_unknown_artist(
        self,
        style: str,
        epoch: str,
        condition: ConditionAnalysis
    ) -> ValuationResult:
        """Estimate value for artwork with unknown artist."""
        # Base value for unknown artist in given style/epoch
        base_values = {
            "Impressionism": 5000,
            "Post-Impressionism": 4000,
            "Expressionism": 3500,
            "Baroque": 6000,
            "Renaissance": 8000,
            "Modern": 2000,
            "Abstract": 2500,
        }

        base_value = base_values.get(epoch, 2000)

        # Adjust for condition
        condition_factor = 1.0 - (condition.damage_score * 0.5)
        estimated_value = base_value * condition_factor

        # Wide range due to uncertainty
        min_value = estimated_value * 0.3
        max_value = estimated_value * 2.0

        return ValuationResult(
            estimated_value=float(estimated_value),
            min=float(min_value),
            max=float(max_value),
            confidence="low",
            rationale=f"Estimated based on {epoch} style without specific artist match. Wide range due to uncertainty.",
            comparable_sales=0
        )

    def _estimate_without_comparables(
        self,
        artist_candidates: List[ArtistCandidate],
        style: str,
        epoch: str,
        condition: ConditionAnalysis,
        authenticity_score: int
    ) -> ValuationResult:
        """Estimate value when no comparable sales are found."""
        # Use artist reputation as proxy
        top_artist = artist_candidates[0]

        # Base values by artist "tier" (simplified)
        if top_artist.similarity > 0.85:
            base_value = 15000  # Strong match to known artist
        elif top_artist.similarity > 0.70:
            base_value = 8000  # Moderate match
        else:
            base_value = 3000  # Weak match

        # Apply adjustments
        adjusted_value = self._apply_adjustments(
            base_value, condition, authenticity_score, top_artist.similarity
        )

        # Wide range due to lack of comparables
        min_value = adjusted_value * 0.4
        max_value = adjusted_value * 2.5

        return ValuationResult(
            estimated_value=float(adjusted_value),
            min=float(min_value),
            max=float(max_value),
            confidence="low",
            rationale=f"Estimated value for {top_artist.name}-style work without direct comparable sales. Expert appraisal recommended.",
            comparable_sales=0
        )


# Global valuation engine instance
valuation_engine = ValuationEngine()
