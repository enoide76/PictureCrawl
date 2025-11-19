"""
Artwork condition analysis using computer vision.

Detects craquelé, yellowing, stains, cracks, and other damage.
"""
from typing import List

import cv2
import numpy as np
from PIL import Image

from backend.core.logging import log
from backend.core.models import ConditionAnalysis


class ConditionAnalyzer:
    """
    Analyzes the physical condition of artwork from images.

    Detects various types of damage and aging indicators.
    """

    def __init__(self):
        """Initialize condition analyzer."""
        pass

    def analyze(self, image: Image.Image) -> ConditionAnalysis:
        """
        Perform complete condition analysis on an artwork image.

        Args:
            image: PIL Image of artwork

        Returns:
            ConditionAnalysis with detected conditions
        """
        log.info("Analyzing artwork condition...")

        # Convert to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Detect various conditions
        has_craquele = self._detect_craquele(img_gray)
        has_yellowing = self._detect_yellowing(img_array)
        has_stains = self._detect_stains(img_bgr)
        cracks = self._detect_cracks(img_gray)

        # Calculate overall damage score
        damage_score = self._calculate_damage_score(
            has_craquele, has_yellowing, has_stains, len(cracks)
        )

        # Generate notes
        notes = self._generate_notes(
            has_craquele, has_yellowing, has_stains, cracks
        )

        result = ConditionAnalysis(
            craquele=has_craquele,
            yellowing=has_yellowing,
            stains=has_stains,
            cracks=cracks,
            damage_score=damage_score,
            notes=notes
        )

        log.info(f"Condition analysis: damage_score={damage_score:.2f}")
        return result

    def _detect_craquele(self, img_gray: np.ndarray) -> bool:
        """
        Detect craquelé (fine cracking pattern) in artwork.

        Args:
            img_gray: Grayscale image

        Returns:
            True if craquelé detected
        """
        # Apply edge detection
        edges = cv2.Canny(img_gray, 30, 100)

        # Look for fine crack patterns
        # Craquelé appears as a network of fine lines

        # Count edge pixels
        edge_density = np.sum(edges > 0) / edges.size

        # Apply morphological operations to detect crack networks
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Craquelé typically creates many small contours
        small_contours = sum(1 for c in contours if cv2.contourArea(c) < 100)

        # Heuristic: high edge density + many small contours
        has_craquele = edge_density > 0.05 and small_contours > 50

        log.debug(f"Craquelé detection: {has_craquele} (edge_density={edge_density:.3f}, small_contours={small_contours})")

        return has_craquele

    def _detect_yellowing(self, img_rgb: np.ndarray) -> bool:
        """
        Detect yellowing (varnish aging) in artwork.

        Args:
            img_rgb: RGB image array

        Returns:
            True if yellowing detected
        """
        # Convert to LAB color space
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        # Extract L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(img_lab)

        # Yellowing shows up as high B values (yellow-blue axis)
        mean_b = np.mean(b_channel)
        std_b = np.std(b_channel)

        # Check for overall yellow cast
        # LAB B channel: negative = blue, positive = yellow
        # Typical range is around 128 (neutral)

        has_yellowing = mean_b > 135 and std_b < 20

        log.debug(f"Yellowing detection: {has_yellowing} (mean_b={mean_b:.1f})")

        return has_yellowing

    def _detect_stains(self, img_bgr: np.ndarray) -> bool:
        """
        Detect stains or discoloration.

        Args:
            img_bgr: BGR image

        Returns:
            True if stains detected
        """
        # Convert to HSV
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Detect brown/dark spots (common stains)
        # Brown stains typically have low saturation and low value

        h, s, v = cv2.split(img_hsv)

        # Find dark spots with low saturation
        dark_mask = (v < 50) & (s < 100)

        # Calculate percentage of dark areas
        dark_percentage = np.sum(dark_mask) / dark_mask.size

        # Also check for color inconsistencies
        color_std = np.std(img_bgr, axis=(0, 1))
        color_variation = np.mean(color_std)

        has_stains = dark_percentage > 0.05 or color_variation > 60

        log.debug(f"Stain detection: {has_stains} (dark_pct={dark_percentage:.3f})")

        return has_stains

    def _detect_cracks(self, img_gray: np.ndarray) -> List[str]:
        """
        Detect and locate significant cracks.

        Args:
            img_gray: Grayscale image

        Returns:
            List of crack locations (e.g., ["upper_left", "center"])
        """
        height, width = img_gray.shape

        # Apply edge detection with higher threshold for major cracks
        edges = cv2.Canny(img_gray, 50, 150)

        # Divide image into regions
        regions = {
            "upper_left": (0, 0, width // 3, height // 3),
            "upper_center": (width // 3, 0, 2 * width // 3, height // 3),
            "upper_right": (2 * width // 3, 0, width, height // 3),
            "center_left": (0, height // 3, width // 3, 2 * height // 3),
            "center": (width // 3, height // 3, 2 * width // 3, 2 * height // 3),
            "center_right": (2 * width // 3, height // 3, width, 2 * height // 3),
            "lower_left": (0, 2 * height // 3, width // 3, height),
            "lower_center": (width // 3, 2 * height // 3, 2 * width // 3, height),
            "lower_right": (2 * width // 3, 2 * height // 3, width, height),
        }

        crack_locations = []

        for region_name, (x1, y1, x2, y2) in regions.items():
            region_edges = edges[y1:y2, x1:x2]
            edge_density = np.sum(region_edges > 0) / region_edges.size

            # If high edge density in region, mark as having cracks
            if edge_density > 0.08:
                crack_locations.append(region_name)

        log.debug(f"Crack locations: {crack_locations}")

        return crack_locations

    def _calculate_damage_score(
        self,
        has_craquele: bool,
        has_yellowing: bool,
        has_stains: bool,
        num_crack_regions: int
    ) -> float:
        """
        Calculate overall damage score (0 = mint, 1 = heavily damaged).

        Args:
            has_craquele: Whether craquelé detected
            has_yellowing: Whether yellowing detected
            has_stains: Whether stains detected
            num_crack_regions: Number of regions with cracks

        Returns:
            Damage score from 0.0 to 1.0
        """
        score = 0.0

        if has_craquele:
            score += 0.15  # Craquelé is often acceptable aging

        if has_yellowing:
            score += 0.20  # Yellowing indicates aging

        if has_stains:
            score += 0.30  # Stains are more serious

        # Cracks are most serious
        crack_penalty = min(0.35, num_crack_regions * 0.05)
        score += crack_penalty

        # Clip to [0, 1]
        score = min(1.0, score)

        return float(score)

    def _generate_notes(
        self,
        has_craquele: bool,
        has_yellowing: bool,
        has_stains: bool,
        crack_locations: List[str]
    ) -> str:
        """
        Generate human-readable condition notes.

        Args:
            has_craquele: Whether craquelé detected
            has_yellowing: Whether yellowing detected
            has_stains: Whether stains detected
            crack_locations: List of crack locations

        Returns:
            Condition description string
        """
        notes = []

        if has_craquele:
            notes.append("Fine craquelé pattern visible (typical aging)")

        if has_yellowing:
            notes.append("Overall yellowing detected (likely varnish aging)")

        if has_stains:
            notes.append("Visible stains or discoloration present")

        if crack_locations:
            locations_str = ", ".join(crack_locations)
            notes.append(f"Significant cracks detected in: {locations_str}")

        if not notes:
            return "Overall good condition. No significant damage detected."

        return ". ".join(notes) + "."


# Global condition analyzer instance
condition_analyzer = ConditionAnalyzer()
