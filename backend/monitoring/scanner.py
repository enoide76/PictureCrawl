"""
Market monitoring scanner - main orchestration module.
"""
from typing import Dict, List, Optional

from backend.core.config import settings
from backend.core.logging import log


def run_monitoring_scan(
    sources: Optional[List[str]] = None,
    max_items: int = 100
) -> Dict:
    """
    Run market monitoring scan across specified sources.

    Args:
        sources: List of sources to scan (None = all enabled sources)
        max_items: Maximum items to process per source

    Returns:
        Dictionary with scan results
    """
    log.info("Starting market monitoring scan...")

    results = {
        "items_scanned": 0,
        "artworks_found": 0,
        "alerts_created": 0
    }

    # If monitoring is disabled, return empty results
    if not settings.MONITORING_ENABLED or settings.MOCK_MARKETPLACES:
        log.info("Using mock marketplace scanner")
        results = _run_mock_scan(max_items)
    else:
        # Real scanning would go here
        log.warning("Real marketplace scanning not yet implemented")

    log.info(f"Scan complete: {results}")
    return results


def _run_mock_scan(max_items: int) -> Dict:
    """
    Run a mock scan using sample data for testing.

    Args:
        max_items: Maximum items to process

    Returns:
        Dictionary with scan results
    """
    import json
    from backend.core.database import db
    from backend.core.models import MarketItem
    from backend.vision.pipeline import vision_pipeline

    # Load mock listings
    mock_data_path = settings.MOCK_MARKETPLACE_DIR / "sample_listings.json"

    if not mock_data_path.exists():
        log.warning("Mock marketplace data not found")
        return {"items_scanned": 0, "artworks_found": 0, "alerts_created": 0}

    with open(mock_data_path, "r") as f:
        listings = json.load(f)

    items_scanned = 0
    artworks_found = 0
    alerts_created = 0

    for listing in listings[:max_items]:
        items_scanned += 1

        # Check if already in database
        try:
            # Create market item
            item = MarketItem(
                source=listing["source"],
                url=listing["url"],
                title=listing["title"],
                description=listing.get("description"),
                price=listing["price"],
                currency=listing.get("currency", "EUR"),
                seller=listing.get("seller"),
                location=listing.get("location"),
                image_urls=listing.get("image_urls", []),
                is_artwork=listing.get("potential_artwork", False)
            )

            # Add to database
            item_id = db.add_market_item(item)

            if item.is_artwork:
                artworks_found += 1

            log.info(f"Processed mock item: {item.title}")

        except Exception as e:
            log.error(f"Error processing item: {e}")

    return {
        "items_scanned": items_scanned,
        "artworks_found": artworks_found,
        "alerts_created": alerts_created
    }
