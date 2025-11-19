"""
Embedding generation using CLIP or similar vision models.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from backend.core.config import settings
from backend.core.logging import log


class EmbeddingGenerator:
    """Generates embeddings for images using vision transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Model identifier (e.g., 'openai/clip-vit-base-patch32')
            device: Device to use ('cuda', 'mps', 'cpu')
        """
        self.model_name = model_name or settings.VISION_MODEL
        self.device = device or settings.DEVICE

        # Auto-detect device if 'auto'
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        log.info(f"Initializing embedding generator with model: {self.model_name}")
        log.info(f"Using device: {self.device}")

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the vision model and processor."""
        try:
            from transformers import CLIPModel, CLIPProcessor

            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )

            self.model = CLIPModel.from_pretrained(
                self.model_name,
                cache_dir=settings.MODEL_CACHE_DIR
            )

            self.model.to(self.device)
            self.model.eval()

            log.info(f"Model loaded successfully: {self.model_name}")

        except Exception as e:
            log.error(f"Failed to load model {self.model_name}: {e}")
            log.warning("Falling back to mock embeddings")
            self.model = None
            self.processor = None

    def generate_embedding(
        self,
        image: Union[Image.Image, str, Path],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding vector for an image.

        Args:
            image: PIL Image or path to image file
            normalize: Whether to normalize the embedding to unit length

        Returns:
            Embedding vector as numpy array
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            from backend.vision.image_processor import processor as img_processor
            image = img_processor.load_image(image)

        # If model is not available, return mock embedding
        if self.model is None or self.processor is None:
            return self._generate_mock_embedding()

        try:
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Convert to numpy
            embedding = image_features.cpu().numpy().squeeze()

            # Normalize if requested
            if normalize:
                embedding = embedding / np.linalg.norm(embedding)

            log.debug(f"Generated embedding with shape: {embedding.shape}")
            return embedding

        except Exception as e:
            log.error(f"Failed to generate embedding: {e}")
            return self._generate_mock_embedding()

    def _generate_mock_embedding(self) -> np.ndarray:
        """
        Generate a mock embedding for testing without models.

        Returns:
            Random normalized vector of dimension settings.EMBEDDING_DIM
        """
        log.warning("Generating mock embedding")
        embedding = np.random.randn(settings.EMBEDDING_DIM)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def generate_text_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for text (useful for text-to-image search).

        Args:
            text: Input text
            normalize: Whether to normalize the embedding

        Returns:
            Embedding vector as numpy array
        """
        if self.model is None or self.processor is None:
            return self._generate_mock_embedding()

        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            embedding = text_features.cpu().numpy().squeeze()

            if normalize:
                embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            log.error(f"Failed to generate text embedding: {e}")
            return self._generate_mock_embedding()

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 > 0:
            embedding1 = embedding1 / norm1
        if norm2 > 0:
            embedding2 = embedding2 / norm2

        similarity = np.dot(embedding1, embedding2)

        # Clip to [0, 1] range
        similarity = np.clip(similarity, 0.0, 1.0)

        return float(similarity)

    def batch_generate_embeddings(
        self,
        images: list[Union[Image.Image, str, Path]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images in batches.

        Args:
            images: List of PIL Images or paths
            batch_size: Batch size for processing

        Returns:
            Array of embeddings (N, embedding_dim)
        """
        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_embeddings = [self.generate_embedding(img) for img in batch]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)


# Global embedding generator instance
embedding_generator = EmbeddingGenerator()
