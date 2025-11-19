# Market Monitoring System Design

## Overview

The Gemäldeagent market monitoring subsystem automatically scans online marketplaces, auction sites, and classifieds platforms to discover potential artworks. It analyzes discovered items, detects duplicates, and creates alerts for interesting finds.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Monitoring Orchestrator                        │
└────────────────────┬─────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼─────┐ ┌───▼──────┐ ┌──▼────────┐
│ Marketplace │ │ Auction  │ │Flea Market│
│  Scanners   │ │ Scanners │ │ Scanners  │
└───────┬─────┘ └────┬─────┘ └─────┬─────┘
        │            │             │
        └────────────┼─────────────┘
                     │
             ┌───────▼────────┐
             │Image Extractor │
             └───────┬────────┘
                     │
             ┌───────▼────────┐
             │Image Detector  │ ──► Is Artwork?
             └───────┬────────┘
                     │ Yes
             ┌───────▼────────┐
             │  Full Analysis │ ──► Vision + Artist + Valuation
             └───────┬────────┘
                     │
             ┌───────▼────────┐
             │ Deduplication  │ ──► Check for duplicates
             └───────┬────────┘
                     │
             ┌───────▼────────┐
             │  Database      │ ──► Store market_items
             └───────┬────────┘
                     │
             ┌───────▼────────┐
             │ Alert Manager  │ ──► Evaluate alert rules
             └───────┬────────┘
                     │
             ┌───────▼────────┐
             │ Notifications  │ ──► Email / Telegram
             └────────────────┘
```

## Components

### 1. Marketplace Scanners

**Location:** `backend/monitoring/marketplace_scanner/`

**Purpose:** Scan various online platforms for artwork listings.

**Base Class:** `BaseMarketplaceScanner`

All marketplace scanners inherit from this abstract base class and implement the `scan()` method.

**Implemented Scanners:**
- `MockMarketplaceScanner`: Reads from local JSON files for testing
- Additional scanners can be added for real marketplaces

**Scanner Interface:**
```python
class BaseMarketplaceScanner(ABC):
    @abstractmethod
    def scan(self, max_items: int = 100) -> List[MarketItem]:
        """Scan marketplace and return items."""
        pass
```

**Creating a New Scanner:**

1. Create a new file in `backend/monitoring/marketplace_scanner/`
2. Inherit from `BaseMarketplaceScanner`
3. Implement the `scan()` method
4. Register in the monitoring orchestrator

Example:
```python
from backend.monitoring.marketplace_scanner.base_scanner import BaseMarketplaceScanner

class EbayScanner(BaseMarketplaceScanner):
    def __init__(self):
        super().__init__("ebay")

    def scan(self, max_items: int = 100) -> List[MarketItem]:
        # Implement eBay API or scraping logic
        items = []
        # ... fetch and parse listings
        return items
```

### 2. Image Extraction & Detection

**Location:** `backend/monitoring/image_extractor/`

**Components:**
- `detector.py`: `ImageDetector` class

**Purpose:** Extract images from listings and determine if they're artworks.

**Process:**
1. Extract image URLs from listing HTML/JSON
2. Download images
3. Run through vision pipeline's `is_artwork()` classifier
4. If artwork detected, perform full analysis

**Configuration:**
```python
# .env
ARTWORK_DETECTION_THRESHOLD=0.5  # Confidence threshold
```

### 3. Deduplication

**Location:** `backend/monitoring/deduplication/`

**Purpose:** Prevent duplicate entries for the same artwork across different platforms.

**Algorithm:**
- Uses perceptual hashing (pHash)
- Computes 64-bit hash of each image
- Compares new images against existing hashes using Hamming distance
- Configurable threshold (default: 5 bits difference)

**Process:**
```python
from backend.monitoring.deduplication.deduplicator import deduplicator

# Compute hash
phash = deduplicator.compute_phash(image)

# Check for duplicates
duplicate_id = deduplicator.is_duplicate(image)

if duplicate_id:
    # Skip this item
    pass
else:
    # Process as new item
    pass
```

**Configuration:**
```env
DEDUPLICATION_ENABLED=true
PERCEPTUAL_HASH_THRESHOLD=5  # Hamming distance threshold
```

### 4. Alert System

**Location:** `backend/monitoring/alerting/`

**Purpose:** Create alerts for interesting market finds.

**Alert Rules:**

1. **Bargain Alert (Critical)**
   - Triggers when: `listed_price < (estimated_value * ALERT_BARGAIN_THRESHOLD)`
   - Default threshold: 40% of estimated value
   - Example: Item listed at €500, estimated value €2,000

2. **Rare Style Alert (Warning)**
   - Triggers for specific epochs/styles
   - Configurable list of rare styles
   - Default: Impressionism, Post-Impressionism, Expressionism

3. **Quality Find Alert (Warning)**
   - Triggers when:
     - Authenticity score ≥ 80
     - Estimated value > €5,000
     - Listed price < estimated value

**Alert Levels:**
- `critical`: High-priority finds (bargains, undervalued works)
- `warning`: Interesting finds (rare styles, quality pieces)
- `info`: General notifications

**Notification Channels:**

Configured in `.env`:

```env
# Email
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_TO=your@email.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email
SMTP_PASSWORD=your_password

# Telegram
ALERT_TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Alert Workflow:**
1. Item analyzed
2. Alert rules evaluated
3. Alerts created in database
4. Notifications sent
5. Alert marked as notified

### 5. Database Schema

**Tables:**

**market_items:**
```sql
id                INTEGER PRIMARY KEY
source            TEXT        -- Platform name (ebay, willhaben, etc.)
url               TEXT UNIQUE -- Item URL
title             TEXT        -- Item title
description       TEXT        -- Item description
price             REAL        -- Listed price
currency          TEXT        -- Currency (EUR, USD, etc.)
seller            TEXT        -- Seller identifier
location          TEXT        -- Item location
timestamp_found   TIMESTAMP   -- When item was discovered
image_hash        TEXT        -- Perceptual hash
analysis_json     TEXT        -- Full analysis result (JSON)
is_artwork        BOOLEAN     -- Artwork classification
```

**market_images:**
```sql
id          INTEGER PRIMARY KEY
item_id     INTEGER     -- Foreign key to market_items
image_url   TEXT        -- Image URL
phash       TEXT        -- Perceptual hash
```

**alerts:**
```sql
id          INTEGER PRIMARY KEY
item_id     INTEGER     -- Foreign key to market_items
alert_level TEXT        -- critical, warning, info
reason      TEXT        -- Human-readable alert reason
created_at  TIMESTAMP   -- When alert was created
notified    BOOLEAN     -- Whether notification was sent
```

## Workflow

### Manual Scan Trigger

Via API:
```bash
POST http://localhost:8000/api/monitoring/run-scan
```

Via Python:
```python
from backend.monitoring.scanner import run_monitoring_scan

results = run_monitoring_scan(
    sources=["mock_marketplace"],
    max_items=100
)
```

### Automated Scanning (Cron/Scheduler)

Create a cron job or use a scheduler:

```bash
# Crontab entry for hourly scans
0 * * * * cd /path/to/gemäldeagent && python -m backend.monitoring.scanner
```

Or use Python scheduler:
```python
import schedule
import time

def job():
    run_monitoring_scan()

schedule.every().hour.do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Configuration

All monitoring settings in `.env`:

```env
# Enable/disable monitoring
MONITORING_ENABLED=true
MONITORING_INTERVAL_MINUTES=60
MONITORING_MAX_ITEMS_PER_SCAN=100

# Marketplace APIs (optional)
EBAY_API_KEY=your_key
WILLHABEN_ENABLED=true

# Scraping behavior
SCRAPING_DELAY_SECONDS=2
SCRAPING_MAX_RETRIES=3
SCRAPING_TIMEOUT_SECONDS=30
USER_AGENT=Mozilla/5.0 (compatible; GemäldeagentBot/1.0)

# Deduplication
DEDUPLICATION_ENABLED=true
PERCEPTUAL_HASH_THRESHOLD=5

# Alerts
ALERT_ENABLED=true
ALERT_BARGAIN_THRESHOLD=0.4
ALERT_MIN_AUTHENTICITY_SCORE=60
ALERT_RARE_STYLES=Impressionism,Post-Impressionism,Expressionism

# Use mock data for testing
MOCK_MARKETPLACES=true
```

## Performance Considerations

1. **Rate Limiting**
   - Respect robots.txt
   - Add delays between requests (SCRAPING_DELAY_SECONDS)
   - Implement exponential backoff for retries

2. **Image Processing**
   - Download images in batches
   - Use async/await for concurrent downloads
   - Cache downloaded images temporarily

3. **Database Optimization**
   - Index on frequently queried fields (source, is_artwork, price)
   - Use connection pooling
   - Implement pagination for large result sets

4. **Scalability**
   - Process items in batches
   - Use message queues (Redis, RabbitMQ) for distributed processing
   - Implement worker pools for parallel scanning

## Security & Legal

1. **Respect Terms of Service**
   - Always check and comply with marketplace ToS
   - Some platforms prohibit scraping

2. **Rate Limiting**
   - Don't overload target servers
   - Implement polite crawling behavior

3. **Data Privacy**
   - Don't store personal seller information unnecessarily
   - Comply with GDPR if operating in EU

4. **Attribution**
   - Clearly attribute data sources
   - Don't misrepresent scraped data as your own

## Extending the System

### Adding New Marketplaces

1. Create scanner class in `marketplace_scanner/`
2. Implement `scan()` method
3. Handle platform-specific HTML/API structure
4. Register in monitoring configuration
5. Test thoroughly with mock data first

### Adding New Alert Rules

1. Edit `alerting/alert_manager.py`
2. Add new evaluation method
3. Call from `evaluate_item()`
4. Update documentation

### Custom Notification Channels

1. Add notification method to `AlertManager`
2. Add configuration in `.env`
3. Implement sending logic
4. Test with sample alerts

## Testing

Use mock marketplace scanner for testing:

```python
from backend.monitoring.marketplace_scanner.mock_scanner import MockMarketplaceScanner

scanner = MockMarketplaceScanner()
items = scanner.scan(max_items=10)

for item in items:
    print(f"{item.title}: €{item.price}")
```

Mock data location: `data/mock_marketplace/sample_listings.json`

## Monitoring Dashboard (Future)

Future enhancements could include:
- Real-time dashboard showing scan status
- Historical charts of found items
- Alert management interface
- Marketplace performance metrics
- Custom alert rule builder
