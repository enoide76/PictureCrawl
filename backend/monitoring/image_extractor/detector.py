"""
Image detection for market monitoring.

Determines if an image is an artwork using the vision pipeline.
"""
from typing import Tuple

from PIL import Image

from backend.core.logging import log
from backend.vision.pipeline import vision_pipeline


class ImageDetector:
    """
    Detects whether an image is an artwork.

    Uses the vision pipeline's is_artwork classification.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize image detector.

        Args:
            threshold: Confidence threshold for artwork classification
        """
        self.threshold = threshold

    def is_artwork(self, image: Image.Image) -> Tuple[bool, float]:
        """
        Determine if an image is an artwork.

        Args:
            image: PIL Image

        Returns:
            Tuple of (is_artwork, confidence)
        """
        try:
            is_art, confidence = vision_pipeline.is_artwork(image, self.threshold)
            log.debug(f"Artwork detection: {is_art} (confidence: {confidence:.2f})")
            return is_art, confidence

        except Exception as e:
            log.error(f"Error in artwork detection: {e}")
            return False, 0.0

    def analyze_if_artwork(self, image: Image.Image) -> dict:
        """
        If image is an artwork, return full analysis.

        Args:
            image: PIL Image

        Returns:
            Dictionary with analysis results or None
        """
        is_art, confidence = self.is_artwork(image)

        if not is_art:
            return {
                "is_artwork": False,
                "artwork_confidence": confidence
            }

        # Perform full vision analysis
        result = vision_pipeline.process_image(image)

        return {
            "is_artwork": True,
            "artwork_confidence": confidence,
            "style_estimation": result["style_estimation"],
            "embedding": result.get("embedding")
        }


# Global image detector instance
image_detector = ImageDetector()
