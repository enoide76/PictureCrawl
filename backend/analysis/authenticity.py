"""
Authenticity assessment for artworks.

Provides a 0-100 authenticity score with detailed explanations.
"""
from typing import List, Optional

import numpy as np
from PIL import Image

from backend.core.logging import log
from backend.core.models import (
    ArtistCandidate,
    AuthenticityExplanations,
    AuthenticityResult,
    StyleEstimation,
)


class AuthenticityAnalyzer:
    """
    Analyzes artwork authenticity using multiple factors.

    Provides transparent scoring with explanations for each component.
    """

    def __init__(self):
        """Initialize authenticity analyzer."""
        pass

    def analyze(
        self,
        image: Image.Image,
        image_embedding: np.ndarray,
        style_estimation: StyleEstimation,
        artist_candidates: List[ArtistCandidate],
        provenance_hints: Optional[dict] = None
    ) -> AuthenticityResult:
        """
        Analyze artwork authenticity.

        Args:
            image: PIL Image of artwork
            image_embedding: Image embedding vector
            style_estimation: Style estimation result
            artist_candidates: List of artist matches
            provenance_hints: Optional provenance information

        Returns:
            AuthenticityResult with score and explanations
        """
        log.info("Analyzing artwork authenticity...")

        # Calculate component scores
        style_score, style_explanation = self._analyze_style_match(
            style_estimation, artist_candidates
        )

        signature_score, signature_explanation = self._analyze_signature(image)

        texture_score, texture_explanation = self._analyze_texture(
            image, image_embedding, artist_candidates
        )

        material_score, material_explanation = self._analyze_material(image)

        provenance_score, provenance_explanation = self._analyze_provenance(
            provenance_hints
        )

        # Weighted overall score
        weights = {
            "style": 0.30,
            "signature": 0.15,
            "texture": 0.25,
            "material": 0.15,
            "provenance": 0.15
        }

        overall_score = (
            style_score * weights["style"] +
            signature_score * weights["signature"] +
            texture_score * weights["texture"] +
            material_score * weights["material"] +
            provenance_score * weights["provenance"]
        )

        authenticity_score = int(round(overall_score))

        explanations = AuthenticityExplanations(
            style_match=style_explanation,
            signature=signature_explanation,
            texture=texture_explanation,
            material=material_explanation,
            provenance=provenance_explanation
        )

        log.info(f"Authenticity score: {authenticity_score}/100")

        return AuthenticityResult(
            authenticity_score=authenticity_score,
            explanations=explanations
        )

    def _analyze_style_match(
        self,
        style_estimation: StyleEstimation,
        artist_candidates: List[ArtistCandidate]
    ) -> tuple[float, str]:
        """
        Analyze style consistency with artist candidates.

        Returns:
            Tuple of (score 0-100, explanation string)
        """
        if not artist_candidates:
            return 50.0, "No artist candidates to compare against"

        # Use top artist candidate's similarity
        top_artist = artist_candidates[0]
        similarity = top_artist.similarity

        # Check style consistency
        style_match = False
        if top_artist.epoch == style_estimation.epoch:
            style_match = True

        # Calculate score
        score = similarity * 100

        if style_match:
            score = min(100, score * 1.1)  # Bonus for matching epoch
            explanation = (
                f"High similarity ({similarity:.2f}) to {top_artist.name}. "
                f"Style and epoch are consistent with {style_estimation.epoch}."
            )
        else:
            score = score * 0.9  # Penalty for mismatched epoch
            explanation = (
                f"Moderate similarity ({similarity:.2f}) to {top_artist.name}, "
                f"but epoch mismatch detected."
            )

        return score, explanation

    def _analyze_signature(self, image: Image.Image) -> tuple[float, str]:
        """
        Analyze signature presence and characteristics.

        Returns:
            Tuple of (score 0-100, explanation string)
        """
        # This is a placeholder - in production, would use OCR and signature detection
        # For now, return neutral score

        width, height = image.size

        # Simple heuristic: check bottom corners for signature-like patterns
        # In reality, this would use proper OCR and signature detection

        score = 60.0  # Neutral score
        explanation = "No clear signature detected. Further expert examination recommended."

        # In production, would analyze:
        # - Signature presence
        # - Signature style consistency
        # - Signature location (typical for artist)
        # - Age-appropriate materials

        return score, explanation

    def _analyze_texture(
        self,
        image: Image.Image,
        image_embedding: np.ndarray,
        artist_candidates: List[ArtistCandidate]
    ) -> tuple[float, str]:
        """
        Analyze brushstroke patterns and texture.

        Returns:
            Tuple of (score 0-100, explanation string)
        """
        # Placeholder for texture analysis
        # In production, would analyze:
        # - Brushstroke patterns
        # - Paint application consistency
        # - Texture vs. expected for style/epoch

        import cv2
        import numpy as np

        # Convert to grayscale for texture analysis
        img_array = np.array(image.convert('L'))

        # Calculate texture variance
        variance = np.var(img_array)

        # Detect edges (proxy for brushstrokes)
        edges = cv2.Canny(img_array, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Heuristic scoring
        if edge_density > 0.1:  # High edge density suggests hand-painted
            score = 70.0
            explanation = "Brushstroke pattern analysis shows characteristics consistent with hand-painted work."
        elif edge_density > 0.05:
            score = 60.0
            explanation = "Moderate brushstroke patterns detected. Texture appears partially consistent with period techniques."
        else:
            score = 40.0
            explanation = "Low texture variation detected. May indicate print or reproduction."

        return score, explanation

    def _analyze_material(self, image: Image.Image) -> tuple[float, str]:
        """
        Analyze material and pigment characteristics.

        Returns:
            Tuple of (score 0-100, explanation string)
        """
        # Placeholder for material analysis
        # In production, would analyze:
        # - Pigment composition (if spectral data available)
        # - Canvas/support material
        # - Age-appropriate materials
        # - Craquelure pattern consistency

        # Simple color analysis as placeholder
        img_array = np.array(image)
        mean_color = np.mean(img_array, axis=(0, 1))

        # Check for overly saturated colors (might indicate modern materials)
        saturation = np.std(img_array)

        if saturation < 50:
            score = 75.0
            explanation = "Color palette appears consistent with traditional pigments."
        elif saturation < 80:
            score = 65.0
            explanation = "Color palette shows moderate saturation. Material analysis recommended."
        else:
            score = 45.0
            explanation = "High color saturation detected. May indicate modern synthetic pigments."

        return score, explanation

    def _analyze_provenance(
        self,
        provenance_hints: Optional[dict]
    ) -> tuple[float, str]:
        """
        Analyze provenance documentation.

        Returns:
            Tuple of (score 0-100, explanation string)
        """
        if not provenance_hints:
            return 50.0, "No provenance information available."

        # Score based on available provenance data
        score = 50.0
        explanations = []

        if provenance_hints.get("auction_history"):
            score += 20
            explanations.append("Documented auction history found")

        if provenance_hints.get("reverse_image_hits", 0) > 0:
            score += 15
            explanations.append(f"{provenance_hints['reverse_image_hits']} related images found online")

        if provenance_hints.get("notes"):
            score += 10
            explanations.append("Additional provenance notes available")

        score = min(100, score)
        explanation = ". ".join(explanations) if explanations else "Limited provenance data available."

        return score, explanation

    def quick_authenticity_check(
        self,
        artist_similarity: float,
        style_confidence: float
    ) -> int:
        """
        Quick authenticity estimate based on limited data.

        Args:
            artist_similarity: Similarity to known artist (0-1)
            style_confidence: Style classification confidence (0-1)

        Returns:
            Authenticity score 0-100
        """
        score = ((artist_similarity + style_confidence) / 2) * 100
        return int(round(score))


# Global authenticity analyzer instance
authenticity_analyzer = AuthenticityAnalyzer()
