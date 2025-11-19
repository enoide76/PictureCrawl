"""
Base marketplace scanner class.

All marketplace-specific scanners inherit from this.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from backend.core.logging import log
from backend.core.models import MarketItem


class BaseMarketplaceScanner(ABC):
    """
    Abstract base class for marketplace scanners.

    Subclasses must implement scan() method.
    """

    def __init__(self, source_name: str):
        """
        Initialize scanner.

        Args:
            source_name: Identifier for this marketplace (e.g., "ebay", "willhaben")
        """
        self.source_name = source_name
        log.info(f"Initialized {self.source_name} scanner")

    @abstractmethod
    def scan(self, max_items: int = 100) -> List[MarketItem]:
        """
        Scan marketplace for items.

        Args:
            max_items: Maximum number of items to retrieve

        Returns:
            List of MarketItem objects
        """
        pass

    def _create_market_item(
        self,
        url: str,
        title: str,
        price: float,
        description: Optional[str] = None,
        currency: str = "EUR",
        seller: Optional[str] = None,
        location: Optional[str] = None,
        image_urls: Optional[List[str]] = None
    ) -> MarketItem:
        """
        Helper to create a MarketItem with standard fields.

        Args:
            url: Item URL
            title: Item title
            price: Price
            description: Optional description
            currency: Currency code
            seller: Seller identifier
            location: Item location
            image_urls: List of image URLs

        Returns:
            MarketItem instance
        """
        return MarketItem(
            source=self.source_name,
            url=url,
            title=title,
            description=description,
            price=price,
            currency=currency,
            seller=seller,
            location=location,
            image_urls=image_urls or [],
            is_artwork=False  # Will be determined by image detection
        )

    def validate_item(self, item: MarketItem) -> bool:
        """
        Validate that an item has required fields.

        Args:
            item: MarketItem to validate

        Returns:
            True if valid
        """
        if not item.url or not item.title or item.price is None:
            log.warning(f"Invalid item: missing required fields")
            return False

        return True
