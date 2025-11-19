"""
Deduplication engine using perceptual hashing.

Prevents duplicate entries for the same artwork across different marketplaces.
"""
from typing import Optional

import imagehash
from PIL import Image

from backend.core.config import settings
from backend.core.database import db
from backend.core.logging import log


class Deduplicator:
    """
    Deduplicates market items using perceptual image hashing.
    """

    def __init__(self, hash_size: int = 8):
        """
        Initialize deduplicator.

        Args:
            hash_size: Size of perceptual hash (default 8 = 64-bit hash)
        """
        self.hash_size = hash_size
        self.threshold = settings.PERCEPTUAL_HASH_THRESHOLD

    def compute_phash(self, image: Image.Image) -> str:
        """
        Compute perceptual hash for an image.

        Args:
            image: PIL Image

        Returns:
            Perceptual hash as hex string
        """
        try:
            # Compute perceptual hash
            phash = imagehash.phash(image, hash_size=self.hash_size)
            return str(phash)

        except Exception as e:
            log.error(f"Error computing perceptual hash: {e}")
            return ""

    def compute_dhash(self, image: Image.Image) -> str:
        """
        Compute difference hash for an image.

        Args:
            image: PIL Image

        Returns:
            Difference hash as hex string
        """
        try:
            dhash = imagehash.dhash(image, hash_size=self.hash_size)
            return str(dhash)

        except Exception as e:
            log.error(f"Error computing difference hash: {e}")
            return ""

    def is_duplicate(self, image: Image.Image) -> Optional[int]:
        """
        Check if an image is a duplicate of an existing item.

        Args:
            image: PIL Image

        Returns:
            Item ID of duplicate if found, None otherwise
        """
        if not settings.DEDUPLICATION_ENABLED:
            return None

        # Compute hash
        phash = self.compute_phash(image)

        if not phash:
            return None

        # Check database for similar hashes
        duplicate_id = db.find_duplicate_by_hash(phash, threshold=self.threshold)

        if duplicate_id:
            log.info(f"Duplicate detected: matches item {duplicate_id}")

        return duplicate_id

    def compute_hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two hashes.

        Args:
            hash1: First hash string
            hash2: Second hash string

        Returns:
            Hamming distance (number of differing bits)
        """
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return h1 - h2

        except Exception as e:
            log.error(f"Error computing Hamming distance: {e}")
            return 999  # Large value indicates no match

    def find_similar_items(
        self,
        image: Image.Image,
        max_distance: int = 10
    ) -> list[tuple[int, int]]:
        """
        Find all similar items within a Hamming distance threshold.

        Args:
            image: PIL Image
            max_distance: Maximum Hamming distance

        Returns:
            List of (item_id, distance) tuples
        """
        phash = self.compute_phash(image)

        if not phash:
            return []

        # In production, would query database efficiently
        # For now, return empty list (would need full implementation)

        return []


# Global deduplicator instance
deduplicator = Deduplicator()
