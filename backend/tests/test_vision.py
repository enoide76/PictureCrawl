"""
Tests for vision pipeline components.
"""
import numpy as np
import pytest
from PIL import Image

from backend.vision.embedding_generator import EmbeddingGenerator
from backend.vision.image_processor import ImageProcessor
from backend.vision.pipeline import VisionPipeline
from backend.vision.style_classifier import StyleClassifier


class TestImageProcessor:
    """Tests for ImageProcessor."""

    def test_create_test_image(self):
        """Create a test image."""
        img = Image.new("RGB", (100, 100), color="red")
        assert img.size == (100, 100)
        assert img.mode == "RGB"

    def test_resize_image(self):
        """Test image resizing."""
        processor = ImageProcessor()
        img = Image.new("RGB", (100, 100), color="blue")

        resized = processor.resize_image(img, (50, 50))
        assert resized.size == (50, 50)

    def test_normalize_image(self):
        """Test image normalization."""
        processor = ImageProcessor()
        img = Image.new("RGB", (10, 10), color=(128, 128, 128))

        normalized = processor.normalize_image(img)
        assert normalized.shape == (10, 10, 3)
        assert 0.0 <= normalized.max() <= 1.0

    def test_to_tensor(self):
        """Test conversion to tensor format."""
        processor = ImageProcessor()
        img = Image.new("RGB", (10, 10), color="green")

        tensor = processor.to_tensor(img)
        assert tensor.shape == (3, 10, 10)  # C, H, W

    def test_extract_color_histogram(self):
        """Test color histogram extraction."""
        processor = ImageProcessor()
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))

        histogram = processor.extract_color_histogram(img, bins=256)
        assert len(histogram) == 256 * 3  # 3 channels

    def test_detect_dominant_colors(self):
        """Test dominant color detection."""
        processor = ImageProcessor()
        img = Image.new("RGB", (100, 100), color=(255, 128, 64))

        dominant = processor.detect_dominant_colors(img, k=3)
        assert dominant.shape == (3, 3)  # k colors, 3 channels


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator."""

    def test_generate_mock_embedding(self):
        """Test mock embedding generation."""
        generator = EmbeddingGenerator()
        embedding = generator._generate_mock_embedding()

        assert len(embedding) > 0
        # Should be normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_generate_embedding_from_image(self):
        """Test embedding generation from image."""
        generator = EmbeddingGenerator()
        img = Image.new("RGB", (224, 224), color="blue")

        embedding = generator.generate_embedding(img)
        assert len(embedding) > 0
        assert isinstance(embedding, np.ndarray)

    def test_compute_similarity(self):
        """Test similarity computation."""
        generator = EmbeddingGenerator()

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])
        emb3 = np.array([0.0, 1.0, 0.0])

        # Same vectors should have similarity 1.0
        sim_same = generator.compute_similarity(emb1, emb2)
        assert abs(sim_same - 1.0) < 0.01

        # Orthogonal vectors should have similarity 0.0
        sim_orthogonal = generator.compute_similarity(emb1, emb3)
        assert abs(sim_orthogonal - 0.0) < 0.01

    def test_generate_text_embedding(self):
        """Test text embedding generation."""
        generator = EmbeddingGenerator()
        embedding = generator.generate_text_embedding("impressionist painting")

        assert len(embedding) > 0
        assert isinstance(embedding, np.ndarray)


class TestStyleClassifier:
    """Tests for StyleClassifier."""

    def test_classifier_has_epochs(self):
        """Test that classifier has epoch definitions."""
        classifier = StyleClassifier()
        assert len(classifier.EPOCHS) > 0
        assert "Impressionism" in classifier.EPOCHS
        assert "Baroque" in classifier.EPOCHS

    def test_classifier_has_styles(self):
        """Test that classifier has style definitions."""
        classifier = StyleClassifier()
        assert len(classifier.STYLES) > 0
        assert "Portrait" in classifier.STYLES
        assert "Landscape" in classifier.STYLES

    def test_classify_epoch(self):
        """Test epoch classification."""
        classifier = StyleClassifier()
        # Use a mock embedding
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)

        results = classifier.classify_epoch(embedding, top_k=3)
        assert len(results) == 3
        assert all(isinstance(epoch, str) and isinstance(score, float) for epoch, score in results)

    def test_classify_style(self):
        """Test style classification."""
        classifier = StyleClassifier()
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)

        results = classifier.classify_style(embedding, top_k=3)
        assert len(results) == 3
        assert all(isinstance(style, str) and isinstance(score, float) for style, score in results)

    def test_is_artwork(self):
        """Test artwork detection."""
        classifier = StyleClassifier()
        img = Image.new("RGB", (224, 224), color="red")

        is_art, confidence = classifier.is_artwork(img)
        assert isinstance(is_art, bool)
        assert 0.0 <= confidence <= 1.0

    def test_classify(self):
        """Test full classification."""
        classifier = StyleClassifier()
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))

        result = classifier.classify(img)
        assert hasattr(result, 'epoch')
        assert hasattr(result, 'style')
        assert hasattr(result, 'confidence')
        assert 0.0 <= result.confidence <= 1.0


class TestVisionPipeline:
    """Tests for VisionPipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = VisionPipeline()
        assert pipeline.processor is not None
        assert pipeline.embedding_generator is not None
        assert pipeline.style_classifier is not None

    def test_process_image(self):
        """Test processing an image through the pipeline."""
        pipeline = VisionPipeline()
        img = Image.new("RGB", (224, 224), color="green")

        result = pipeline.process_image(img)

        assert "image_path" in result
        assert "image_size" in result
        assert "embedding" in result
        assert "is_artwork" in result
        assert "style_estimation" in result

    def test_process_image_minimal(self):
        """Test processing with minimal features."""
        pipeline = VisionPipeline()
        img = Image.new("RGB", (100, 100), color="blue")

        result = pipeline.process_image(
            img,
            include_embeddings=False,
            include_style=False,
            include_artwork_detection=False
        )

        assert "image_path" in result
        assert "embedding" not in result
        assert "is_artwork" not in result
        assert "style_estimation" not in result

    def test_classify_style_convenience(self):
        """Test style classification convenience method."""
        pipeline = VisionPipeline()
        img = Image.new("RGB", (224, 224), color="red")

        style_est = pipeline.classify_style(img)
        assert hasattr(style_est, 'epoch')
        assert hasattr(style_est, 'style')
        assert hasattr(style_est, 'confidence')

    def test_is_artwork_convenience(self):
        """Test artwork detection convenience method."""
        pipeline = VisionPipeline()
        img = Image.new("RGB", (224, 224), color="yellow")

        is_art, confidence = pipeline.is_artwork(img)
        assert isinstance(is_art, bool)
        assert 0.0 <= confidence <= 1.0

    def test_extract_features(self):
        """Test feature extraction."""
        pipeline = VisionPipeline()
        img = Image.new("RGB", (100, 100), color=(128, 64, 32))

        features = pipeline.extract_features(img)

        assert "size" in features
        assert "mode" in features
        assert "color_histogram" in features
        assert "dominant_colors" in features
        assert "embedding" in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
