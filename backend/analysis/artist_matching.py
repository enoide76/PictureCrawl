"""
Artist matching using embedding similarity search with FAISS.
"""
import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from backend.core.config import settings
from backend.core.logging import log
from backend.core.models import ArtistCandidate
from backend.vision.embedding_generator import embedding_generator


class ArtistMatcher:
    """
    Matches artworks to known artists using embedding similarity.

    Uses FAISS for efficient nearest neighbor search in embedding space.
    """

    def __init__(
        self,
        metadata_path: Optional[Path] = None,
        index_path: Optional[Path] = None
    ):
        """
        Initialize artist matcher.

        Args:
            metadata_path: Path to artist metadata JSON file
            index_path: Path to FAISS index file (optional, will be created if not exists)
        """
        self.metadata_path = metadata_path or (
            settings.EMBEDDINGS_DIR / "artist_metadata.json"
        )
        self.index_path = index_path or (
            settings.EMBEDDINGS_DIR / "artist_embeddings.faiss"
        )

        self.artists = []
        self.index = None
        self.embedding_dim = settings.EMBEDDING_DIM

        self._load_metadata()
        self._load_or_create_index()

    def _load_metadata(self):
        """Load artist metadata from JSON file."""
        if not self.metadata_path.exists():
            log.warning(f"Artist metadata not found at {self.metadata_path}")
            self.artists = []
            return

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.artists = json.load(f)

        log.info(f"Loaded {len(self.artists)} artist records")

    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                log.info(f"Loaded FAISS index from {self.index_path}")
                return
            except Exception as e:
                log.warning(f"Failed to load FAISS index: {e}")

        # Create new index
        log.info("Creating new FAISS index...")
        self._create_index()

    def _create_index(self):
        """Create FAISS index from artist metadata."""
        if not self.artists:
            log.warning("No artists loaded, creating empty index")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            return

        # Generate embeddings for each artist's representative works
        embeddings = []

        for artist in self.artists:
            # Generate embedding from text description
            text = (
                f"{artist['artist_name']} painting in {artist['style']} style, "
                f"{artist['epoch']} period, {artist['signature_style']}"
            )

            embedding = embedding_generator.generate_text_embedding(text)
            embeddings.append(embedding)

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings_array)

        # Save index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        log.info(f"Created FAISS index with {len(embeddings)} artist embeddings")

    def match_artist(
        self,
        image_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[ArtistCandidate]:
        """
        Find artists most similar to the given image embedding.

        Args:
            image_embedding: Image embedding vector
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of ArtistCandidate objects
        """
        if self.index is None or len(self.artists) == 0:
            log.warning("No artist index available")
            return []

        # Ensure embedding is the right shape
        if len(image_embedding.shape) == 1:
            query_embedding = image_embedding.reshape(1, -1).astype(np.float32)
        else:
            query_embedding = image_embedding.astype(np.float32)

        # Search in FAISS index
        k = min(top_k, len(self.artists))
        distances, indices = self.index.search(query_embedding, k)

        # Convert L2 distances to similarities
        # L2 distance to cosine similarity approximation
        # For normalized vectors: similarity ≈ 1 - (distance^2 / 2)
        distances = distances[0]  # Get first query result
        indices = indices[0]

        candidates = []

        for idx, distance in zip(indices, distances):
            if idx == -1:  # Invalid index
                continue

            artist = self.artists[idx]

            # Convert L2 distance to similarity (approximate)
            # For normalized vectors, cosine_sim ≈ 1 - (L2_dist^2 / 2)
            similarity = max(0.0, 1.0 - (distance / 2.0))

            if similarity < min_similarity:
                continue

            candidate = ArtistCandidate(
                name=artist["artist_name"],
                similarity=float(similarity),
                epoch=artist.get("epoch"),
                style=artist.get("style")
            )

            candidates.append(candidate)

        log.info(f"Found {len(candidates)} artist matches")
        return candidates

    def match_by_style_and_epoch(
        self,
        epoch: str,
        style: str,
        max_results: int = 10
    ) -> List[dict]:
        """
        Find artists matching specific epoch and style.

        Args:
            epoch: Art epoch (e.g., "Impressionism")
            style: Art style
            max_results: Maximum number of results

        Returns:
            List of artist metadata dictionaries
        """
        matches = []

        for artist in self.artists:
            if artist.get("epoch") == epoch or artist.get("style") == style:
                matches.append(artist)

            if len(matches) >= max_results:
                break

        log.info(f"Found {len(matches)} artists for {epoch}/{style}")
        return matches

    def get_artist_by_name(self, name: str) -> Optional[dict]:
        """
        Get artist metadata by name.

        Args:
            name: Artist name

        Returns:
            Artist metadata dictionary or None
        """
        for artist in self.artists:
            if artist["artist_name"].lower() == name.lower():
                return artist

        return None

    def add_artist(
        self,
        artist_data: dict,
        embedding: Optional[np.ndarray] = None
    ):
        """
        Add a new artist to the database.

        Args:
            artist_data: Artist metadata dictionary
            embedding: Optional pre-computed embedding
        """
        # Add to metadata
        self.artists.append(artist_data)

        # Save updated metadata
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.artists, f, indent=2, ensure_ascii=False)

        # Regenerate index
        self._create_index()

        log.info(f"Added artist: {artist_data['artist_name']}")

    def rebuild_index(self):
        """Rebuild the FAISS index from current metadata."""
        log.info("Rebuilding artist index...")
        self._create_index()


# Global artist matcher instance
artist_matcher = ArtistMatcher()
