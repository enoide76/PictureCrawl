"""
Image preprocessing and normalization.
"""
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image

from backend.core.logging import log


class ImageProcessor:
    """Handles image loading, preprocessing, and normalization."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image processor.

        Args:
            target_size: Target size for image resizing (width, height)
        """
        self.target_size = target_size

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load image from file path.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file is not a valid image
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path)
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            log.debug(f"Loaded image: {image_path} ({image.size})")
            return image

        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")

    def load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """
        Load image from bytes.

        Args:
            image_bytes: Image data as bytes

        Returns:
            PIL Image
        """
        from io import BytesIO

        image = Image.open(BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def resize_image(self, image: Image.Image, size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Resize image to target size.

        Args:
            image: PIL Image
            size: Target size (width, height), uses self.target_size if None

        Returns:
            Resized PIL Image
        """
        if size is None:
            size = self.target_size

        return image.resize(size, Image.Resampling.LANCZOS)

    def normalize_image(self, image: Image.Image) -> np.ndarray:
        """
        Normalize image to [0, 1] range.

        Args:
            image: PIL Image

        Returns:
            Normalized numpy array (H, W, 3)
        """
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        return img_array

    def to_tensor(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to tensor format (C, H, W).

        Args:
            image: PIL Image

        Returns:
            Numpy array in (C, H, W) format
        """
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        # Convert from (H, W, C) to (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array

    def preprocess_for_clip(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for CLIP model.

        Args:
            image: PIL Image

        Returns:
            Preprocessed numpy array
        """
        # Resize to 224x224 (CLIP default)
        image = self.resize_image(image, (224, 224))

        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0

        # CLIP normalization (ImageNet stats)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])

        img_array = (img_array - mean) / std

        # Convert to (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))

        return img_array

    def load_as_cv2(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image as OpenCV numpy array for computer vision operations.

        Args:
            image_path: Path to image file

        Returns:
            OpenCV image (BGR format)
        """
        image_path = str(image_path)
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Failed to load image with OpenCV: {image_path}")

        log.debug(f"Loaded CV2 image: {image_path} ({img.shape})")
        return img

    def pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV format.

        Args:
            pil_image: PIL Image

        Returns:
            OpenCV image (BGR format)
        """
        # Convert RGB to BGR
        rgb_array = np.array(pil_image)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array

    def cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image to PIL Image.

        Args:
            cv2_image: OpenCV image (BGR format)

        Returns:
            PIL Image
        """
        # Convert BGR to RGB
        rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_array)

    def extract_color_histogram(self, image: Image.Image, bins: int = 256) -> np.ndarray:
        """
        Extract color histogram features.

        Args:
            image: PIL Image
            bins: Number of histogram bins per channel

        Returns:
            Flattened histogram array
        """
        img_array = np.array(image)

        histograms = []
        for channel in range(3):  # R, G, B
            hist, _ = np.histogram(img_array[:, :, channel], bins=bins, range=(0, 256))
            # Normalize
            hist = hist / hist.sum()
            histograms.append(hist)

        return np.concatenate(histograms)

    def detect_dominant_colors(self, image: Image.Image, k: int = 5) -> np.ndarray:
        """
        Detect dominant colors using k-means clustering.

        Args:
            image: PIL Image
            k: Number of dominant colors

        Returns:
            Array of dominant colors (k, 3)
        """
        from sklearn.cluster import KMeans

        # Resize for faster processing
        small_image = image.resize((100, 100))
        img_array = np.array(small_image).reshape(-1, 3)

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(img_array)

        return kmeans.cluster_centers_


# Global processor instance
processor = ImageProcessor()
