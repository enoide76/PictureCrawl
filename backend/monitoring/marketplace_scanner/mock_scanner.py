"""
Mock marketplace scanner for testing.

Uses sample data from data/mock_marketplace/.
"""
import json
from pathlib import Path
from typing import List

from backend.core.config import settings
from backend.core.logging import log
from backend.core.models import MarketItem
from backend.monitoring.marketplace_scanner.base_scanner import BaseMarketplaceScanner


class MockMarketplaceScanner(BaseMarketplaceScanner):
    """
    Mock scanner that reads from local JSON files.

    Useful for testing without hitting real marketplaces.
    """

    def __init__(self, data_file: str = "sample_listings.json"):
        """
        Initialize mock scanner.

        Args:
            data_file: JSON file with mock listings
        """
        super().__init__("mock_marketplace")
        self.data_file = settings.MOCK_MARKETPLACE_DIR / data_file

    def scan(self, max_items: int = 100) -> List[MarketItem]:
        """
        Scan mock marketplace data.

        Args:
            max_items: Maximum items to return

        Returns:
            List of MarketItem objects
        """
        log.info(f"Scanning mock marketplace (max: {max_items})")

        if not self.data_file.exists():
            log.warning(f"Mock data file not found: {self.data_file}")
            return []

        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                listings = json.load(f)

            items = []

            for listing in listings[:max_items]:
                item = self._create_market_item(
                    url=listing["url"],
                    title=listing["title"],
                    description=listing.get("description"),
                    price=listing["price"],
                    currency=listing.get("currency", "EUR"),
                    seller=listing.get("seller"),
                    location=listing.get("location"),
                    image_urls=listing.get("image_urls", [])
                )

                # Set potential_artwork from mock data
                item.is_artwork = listing.get("potential_artwork", False)

                if self.validate_item(item):
                    items.append(item)

            log.info(f"Found {len(items)} items in mock marketplace")
            return items

        except Exception as e:
            log.error(f"Error reading mock marketplace data: {e}")
            return []
