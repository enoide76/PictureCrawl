"""
Intensive tests for real model loading and inference.
These tests download and use actual models, making them slower but more realistic.
"""
import os
import pytest
from PIL import Image

from backend.vision.embedding_generator import EmbeddingGenerator
from backend.vision.style_classifier import StyleClassifier
from backend.vision.pipeline import VisionPipeline


@pytest.mark.slow
class TestRealModelLoading:
    """Tests that use real model loading (slow but thorough)."""

    def test_download_and_load_clip_model(self):
        """Test downloading and loading the actual CLIP model."""
        # This will attempt to download the model from HuggingFace
        generator = EmbeddingGenerator()

        # Create a realistic test image
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))

        # Generate embedding - this should use real model if available
        embedding = generator.generate_embedding(img)

        assert embedding is not None
        assert len(embedding) > 0
        print(f"✓ Generated embedding with shape: {embedding.shape}")
        print(f"✓ Using model: {generator.model_name}")
        print(f"✓ Model loaded: {generator.model is not None}")

    def test_real_style_classification(self):
        """Test style classification with real models."""
        classifier = StyleClassifier()

        # Create test images with different characteristics
        test_cases = [
            (Image.new("RGB", (224, 224), color=(255, 0, 0)), "red_image"),
            (Image.new("RGB", (224, 224), color=(0, 0, 255)), "blue_image"),
            (Image.new("RGB", (224, 224), color=(128, 128, 64)), "olive_image"),
        ]

        for img, name in test_cases:
            result = classifier.classify(img)
            print(f"\n{name}:")
            print(f"  Epoch: {result.epoch}")
            print(f"  Style: {result.style}")
            print(f"  Confidence: {result.confidence:.3f}")

            assert result.epoch is not None
            assert result.style is not None
            assert 0.0 <= result.confidence <= 1.0

    def test_full_pipeline_intensive(self):
        """Test the full vision pipeline with all features enabled."""
        pipeline = VisionPipeline()

        # Create a more complex test image
        img = Image.new("RGB", (512, 512))
        pixels = img.load()

        # Create a gradient pattern
        for i in range(512):
            for j in range(512):
                pixels[i, j] = (i % 256, j % 256, (i + j) % 256)

        # Process with all features
        result = pipeline.process_image(
            img,
            include_embeddings=True,
            include_style=True,
            include_artwork_detection=True
        )

        print("\nFull pipeline results:")
        print(f"  Image size: {result['image_size']}")
        print(f"  Has embedding: {'embedding' in result}")
        print(f"  Is artwork: {result.get('is_artwork')}")
        print(f"  Artwork confidence: {result.get('artwork_confidence', 0):.3f}")
        if 'style_estimation' in result:
            style = result['style_estimation']
            print(f"  Style epoch: {style.epoch}")
            print(f"  Style type: {style.style}")
            print(f"  Style confidence: {style.confidence:.3f}")

        assert 'image_size' in result
        assert 'embedding' in result
        assert 'is_artwork' in result
        assert 'style_estimation' in result

    def test_text_to_image_similarity(self):
        """Test text-to-image similarity computation."""
        generator = EmbeddingGenerator()

        # Create test images
        painting_like = Image.new("RGB", (224, 224), color=(139, 90, 60))
        photo_like = Image.new("RGB", (224, 224), color=(200, 200, 200))

        # Generate embeddings
        painting_emb = generator.generate_embedding(painting_like)
        photo_emb = generator.generate_embedding(photo_like)

        # Generate text embeddings
        painting_text_emb = generator.generate_text_embedding("oil painting")
        photo_text_emb = generator.generate_text_embedding("photograph")

        # Compute similarities
        painting_to_painting_text = generator.compute_similarity(painting_emb, painting_text_emb)
        painting_to_photo_text = generator.compute_similarity(painting_emb, photo_text_emb)
        photo_to_photo_text = generator.compute_similarity(photo_emb, photo_text_emb)
        photo_to_painting_text = generator.compute_similarity(photo_emb, painting_text_emb)

        print("\nText-to-image similarity scores:")
        print(f"  Painting-like → 'oil painting': {painting_to_painting_text:.3f}")
        print(f"  Painting-like → 'photograph': {painting_to_photo_text:.3f}")
        print(f"  Photo-like → 'photograph': {photo_to_photo_text:.3f}")
        print(f"  Photo-like → 'oil painting': {photo_to_painting_text:.3f}")

        # All similarities should be valid
        assert 0.0 <= painting_to_painting_text <= 1.0
        assert 0.0 <= painting_to_photo_text <= 1.0
        assert 0.0 <= photo_to_photo_text <= 1.0
        assert 0.0 <= photo_to_painting_text <= 1.0

    def test_batch_embedding_generation(self):
        """Test batch embedding generation for efficiency."""
        generator = EmbeddingGenerator()

        # Create multiple test images
        images = [
            Image.new("RGB", (224, 224), color=(i * 30 % 256, i * 50 % 256, i * 70 % 256))
            for i in range(10)
        ]

        # Generate embeddings in batch
        embeddings = generator.batch_generate_embeddings(images, batch_size=5)

        print(f"\nBatch embedding generation:")
        print(f"  Number of images: {len(images)}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Expected shape: ({len(images)}, embedding_dim)")

        assert embeddings.shape[0] == len(images)
        assert len(embeddings.shape) == 2


if __name__ == "__main__":
    # Run intensive tests
    pytest.main([__file__, "-v", "-s", "-m", "slow"])
