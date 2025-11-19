"""
Main vision pipeline integrating all vision components.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from backend.core.logging import log
from backend.core.models import StyleEstimation
from backend.vision.embedding_generator import embedding_generator
from backend.vision.image_processor import processor
from backend.vision.style_classifier import style_classifier


class VisionPipeline:
    """
    Complete vision pipeline for artwork analysis.

    Orchestrates image loading, preprocessing, embedding generation,
    style classification, and artwork detection.
    """

    def __init__(self):
        """Initialize vision pipeline."""
        self.processor = processor
        self.embedding_generator = embedding_generator
        self.style_classifier = style_classifier
        log.info("Vision pipeline initialized")

    def process_image(
        self,
        image: Union[str, Path, Image.Image, bytes],
        include_embeddings: bool = True,
        include_style: bool = True,
        include_artwork_detection: bool = True
    ) -> Dict:
        """
        Process an image through the complete vision pipeline.

        Args:
            image: Image as file path, PIL Image, or bytes
            include_embeddings: Whether to generate embeddings
            include_style: Whether to perform style classification
            include_artwork_detection: Whether to detect if it's an artwork

        Returns:
            Dictionary with all vision analysis results
        """
        log.info(f"Processing image through vision pipeline")

        # Load image
        if isinstance(image, bytes):
            pil_image = self.processor.load_image_from_bytes(image)
            image_path = "uploaded_image"
        elif isinstance(image, (str, Path)):
            pil_image = self.processor.load_image(image)
            image_path = str(image)
        else:
            pil_image = image
            image_path = "pil_image"

        result = {
            "image_path": image_path,
            "image_size": pil_image.size,
        }

        # Generate embedding
        embedding = None
        if include_embeddings:
            embedding = self.embedding_generator.generate_embedding(pil_image)
            result["embedding"] = embedding
            result["embedding_dim"] = len(embedding)
            log.debug(f"Generated embedding with dimension {len(embedding)}")

        # Artwork detection
        if include_artwork_detection:
            is_art, art_confidence = self.style_classifier.is_artwork(pil_image)
            result["is_artwork"] = is_art
            result["artwork_confidence"] = art_confidence
            log.info(f"Artwork detection: {is_art} (confidence: {art_confidence:.2f})")

        # Style classification
        if include_style:
            style_estimation = self.style_classifier.classify(pil_image)
            result["style_estimation"] = style_estimation
            log.info(f"Style: {style_estimation.epoch} / {style_estimation.style}")

        return result

    def load_and_embed(self, image_path: Union[str, Path]) -> Tuple[Image.Image, np.ndarray]:
        """
        Convenience method to load image and generate embedding.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (PIL Image, embedding array)
        """
        pil_image = self.processor.load_image(image_path)
        embedding = self.embedding_generator.generate_embedding(pil_image)
        return pil_image, embedding

    def classify_style(self, image: Union[str, Path, Image.Image]) -> StyleEstimation:
        """
        Convenience method to classify style of an image.

        Args:
            image: Image as file path or PIL Image

        Returns:
            StyleEstimation object
        """
        if isinstance(image, (str, Path)):
            image = self.processor.load_image(image)

        return self.style_classifier.classify(image)

    def is_artwork(
        self,
        image: Union[str, Path, Image.Image],
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Convenience method to detect if image is an artwork.

        Args:
            image: Image as file path or PIL Image
            threshold: Confidence threshold

        Returns:
            Tuple of (is_artwork, confidence)
        """
        if isinstance(image, (str, Path)):
            image = self.processor.load_image(image)

        return self.style_classifier.is_artwork(image, threshold)

    def batch_process(
        self,
        image_paths: list[Union[str, Path]],
        **kwargs
    ) -> list[Dict]:
        """
        Process multiple images in batch.

        Args:
            image_paths: List of image paths
            **kwargs: Additional arguments passed to process_image

        Returns:
            List of result dictionaries
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.process_image(image_path, **kwargs)
                results.append(result)
            except Exception as e:
                log.error(f"Failed to process {image_path}: {e}")
                results.append({
                    "image_path": str(image_path),
                    "error": str(e)
                })

        return results

    def extract_features(self, image: Union[str, Path, Image.Image]) -> Dict:
        """
        Extract comprehensive features from an image.

        Args:
            image: Image as file path or PIL Image

        Returns:
            Dictionary with various image features
        """
        if isinstance(image, (str, Path)):
            pil_image = self.processor.load_image(image)
        else:
            pil_image = image

        features = {}

        # Basic info
        features["size"] = pil_image.size
        features["mode"] = pil_image.mode

        # Color histogram
        features["color_histogram"] = self.processor.extract_color_histogram(pil_image)

        # Dominant colors
        features["dominant_colors"] = self.processor.detect_dominant_colors(pil_image, k=5)

        # Embedding
        features["embedding"] = self.embedding_generator.generate_embedding(pil_image)

        return features


# Global vision pipeline instance
vision_pipeline = VisionPipeline()
