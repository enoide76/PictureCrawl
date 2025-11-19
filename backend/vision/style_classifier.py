"""
Style and epoch classification for artworks.
"""
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from backend.core.logging import log
from backend.core.models import StyleEstimation
from backend.vision.embedding_generator import embedding_generator


class StyleClassifier:
    """Classifies artwork style and epoch using embeddings and heuristics."""

    # Predefined art epochs and styles with representative text descriptions
    EPOCHS = {
        "Renaissance": [
            "renaissance painting",
            "classical italian art",
            "renaissance portrait",
            "religious renaissance art"
        ],
        "Baroque": [
            "baroque painting",
            "dramatic lighting baroque",
            "baroque religious art",
            "chiaroscuro painting"
        ],
        "Rococo": [
            "rococo painting",
            "delicate rococo art",
            "pastel rococo colors",
            "ornate rococo style"
        ],
        "Neoclassicism": [
            "neoclassical painting",
            "classical greek style",
            "heroic neoclassical art"
        ],
        "Romanticism": [
            "romantic painting",
            "dramatic romantic landscape",
            "romantic era art",
            "emotional romantic painting"
        ],
        "Realism": [
            "realistic painting",
            "realist art movement",
            "realistic portrait",
            "naturalistic painting"
        ],
        "Impressionism": [
            "impressionist painting",
            "impressionist landscape",
            "light impressionist brushstrokes",
            "monet style painting"
        ],
        "Post-Impressionism": [
            "post-impressionist painting",
            "van gogh style",
            "cezanne style painting",
            "bold post-impressionist colors"
        ],
        "Expressionism": [
            "expressionist painting",
            "emotional expressionist art",
            "distorted expressionist forms",
            "munch style painting"
        ],
        "Cubism": [
            "cubist painting",
            "geometric cubist art",
            "picasso style",
            "abstract cubist forms"
        ],
        "Surrealism": [
            "surrealist painting",
            "dreamlike surrealist art",
            "dali style painting",
            "surrealist imagery"
        ],
        "Abstract": [
            "abstract painting",
            "non-representational art",
            "abstract expressionism",
            "colorful abstract art"
        ],
        "Modern": [
            "modern art painting",
            "contemporary painting",
            "20th century art"
        ]
    }

    STYLES = {
        "Portrait": ["portrait painting", "face portrait", "portrait art"],
        "Landscape": ["landscape painting", "scenic landscape", "nature landscape"],
        "Still Life": ["still life painting", "still life flowers", "still life objects"],
        "Abstract": ["abstract art", "non-representational painting"],
        "Religious": ["religious painting", "religious art", "biblical scene"],
        "Mythology": ["mythological painting", "classical mythology art"],
        "Genre Scene": ["genre painting", "everyday life scene", "genre art"],
        "Seascape": ["seascape painting", "ocean painting", "maritime art"],
        "Cityscape": ["cityscape painting", "urban scene", "architectural painting"]
    }

    def __init__(self):
        """Initialize style classifier."""
        self.epoch_embeddings: Dict[str, np.ndarray] = {}
        self.style_embeddings: Dict[str, np.ndarray] = {}
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Precompute embeddings for epoch and style descriptions."""
        log.info("Precomputing epoch and style embeddings...")

        # Precompute epoch embeddings
        for epoch, descriptions in self.EPOCHS.items():
            embeddings = []
            for desc in descriptions:
                emb = embedding_generator.generate_text_embedding(desc)
                embeddings.append(emb)

            # Average the embeddings for this epoch
            self.epoch_embeddings[epoch] = np.mean(embeddings, axis=0)

        # Precompute style embeddings
        for style, descriptions in self.STYLES.items():
            embeddings = []
            for desc in descriptions:
                emb = embedding_generator.generate_text_embedding(desc)
                embeddings.append(emb)

            self.style_embeddings[style] = np.mean(embeddings, axis=0)

        log.info(f"Precomputed embeddings for {len(self.epoch_embeddings)} epochs and {len(self.style_embeddings)} styles")

    def classify_epoch(self, image_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify the epoch of an artwork.

        Args:
            image_embedding: Image embedding vector
            top_k: Number of top results to return

        Returns:
            List of (epoch, confidence) tuples
        """
        similarities = {}

        for epoch, epoch_embedding in self.epoch_embeddings.items():
            similarity = embedding_generator.compute_similarity(image_embedding, epoch_embedding)
            similarities[epoch] = similarity

        # Sort by similarity
        sorted_epochs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return sorted_epochs[:top_k]

    def classify_style(self, image_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify the style of an artwork.

        Args:
            image_embedding: Image embedding vector
            top_k: Number of top results to return

        Returns:
            List of (style, confidence) tuples
        """
        similarities = {}

        for style, style_embedding in self.style_embeddings.items():
            similarity = embedding_generator.compute_similarity(image_embedding, style_embedding)
            similarities[style] = similarity

        # Sort by similarity
        sorted_styles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return sorted_styles[:top_k]

    def classify(self, image: Image.Image) -> StyleEstimation:
        """
        Classify both epoch and style for an artwork image.

        Args:
            image: PIL Image of artwork

        Returns:
            StyleEstimation with epoch, style, and confidence
        """
        # Generate image embedding
        image_embedding = embedding_generator.generate_embedding(image)

        # Classify epoch
        top_epochs = self.classify_epoch(image_embedding, top_k=1)
        epoch, epoch_confidence = top_epochs[0]

        # Classify style
        top_styles = self.classify_style(image_embedding, top_k=1)
        style, style_confidence = top_styles[0]

        # Overall confidence is the average
        overall_confidence = (epoch_confidence + style_confidence) / 2

        log.info(f"Classified as {epoch} / {style} (confidence: {overall_confidence:.2f})")

        return StyleEstimation(
            epoch=epoch,
            style=style,
            confidence=float(overall_confidence)
        )

    def is_artwork(self, image: Image.Image, threshold: float = 0.5) -> Tuple[bool, float]:
        """
        Determine if an image is an artwork (painting/drawing) vs. photograph or other.

        Args:
            image: PIL Image
            threshold: Confidence threshold for classification

        Returns:
            Tuple of (is_artwork, confidence)
        """
        # Generate image embedding
        image_embedding = embedding_generator.generate_embedding(image)

        # Compare against "artwork" vs "photograph" text embeddings
        artwork_texts = [
            "painting",
            "oil painting",
            "watercolor painting",
            "drawing",
            "sketch",
            "artwork",
            "fine art"
        ]

        non_artwork_texts = [
            "photograph",
            "photo",
            "digital image",
            "print",
            "poster"
        ]

        # Compute similarities
        artwork_similarities = []
        for text in artwork_texts:
            text_emb = embedding_generator.generate_text_embedding(text)
            sim = embedding_generator.compute_similarity(image_embedding, text_emb)
            artwork_similarities.append(sim)

        non_artwork_similarities = []
        for text in non_artwork_texts:
            text_emb = embedding_generator.generate_text_embedding(text)
            sim = embedding_generator.compute_similarity(image_embedding, text_emb)
            non_artwork_similarities.append(sim)

        avg_artwork_sim = np.mean(artwork_similarities)
        avg_non_artwork_sim = np.mean(non_artwork_similarities)

        # Calculate confidence
        confidence = avg_artwork_sim / (avg_artwork_sim + avg_non_artwork_sim)

        is_art = confidence >= threshold

        log.debug(f"Artwork detection: {is_art} (confidence: {confidence:.2f})")

        return is_art, float(confidence)


# Global style classifier instance
style_classifier = StyleClassifier()
