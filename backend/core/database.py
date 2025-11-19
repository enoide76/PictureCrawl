"""
Database initialization and management.
Uses SQLite for relational data and DuckDB for embeddings.
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from backend.core.config import settings
from backend.core.logging import log
from backend.core.models import Alert, MarketItem


class Database:
    """Database manager for market items and alerts."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection."""
        if db_path is None:
            db_path = settings.DATABASE_URL.replace("sqlite:///", "")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        log.info(f"Database initialized at {self.db_path}")

    def _init_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Market items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                description TEXT,
                price REAL NOT NULL,
                currency TEXT DEFAULT 'EUR',
                seller TEXT,
                location TEXT,
                timestamp_found TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_hash TEXT,
                analysis_json TEXT,
                is_artwork BOOLEAN DEFAULT 0
            )
        """)

        # Market images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                image_url TEXT NOT NULL,
                phash TEXT,
                FOREIGN KEY (item_id) REFERENCES market_items (id) ON DELETE CASCADE
            )
        """)

        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                alert_level TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notified BOOLEAN DEFAULT 0,
                FOREIGN KEY (item_id) REFERENCES market_items (id) ON DELETE CASCADE
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_items_source ON market_items(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_items_is_artwork ON market_items(is_artwork)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_items_price ON market_items(price)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_images_phash ON market_images(phash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_item_id ON alerts(item_id)")

        self.conn.commit()
        log.info("Database tables initialized")

    def add_market_item(self, item: MarketItem) -> int:
        """Add a market item to the database."""
        cursor = self.conn.cursor()

        analysis_json = None
        if item.analysis_result:
            analysis_json = item.analysis_result.model_dump_json()

        try:
            cursor.execute("""
                INSERT INTO market_items (
                    source, url, title, description, price, currency,
                    seller, location, image_hash, analysis_json, is_artwork
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.source, item.url, item.title, item.description,
                item.price, item.currency, item.seller, item.location,
                item.image_hash, analysis_json, item.is_artwork
            ))

            item_id = cursor.lastrowid

            # Add images
            for image_url in item.image_urls:
                cursor.execute("""
                    INSERT INTO market_images (item_id, image_url)
                    VALUES (?, ?)
                """, (item_id, image_url))

            self.conn.commit()
            log.info(f"Added market item {item_id}: {item.title}")
            return item_id

        except sqlite3.IntegrityError:
            log.warning(f"Market item already exists: {item.url}")
            # Return existing ID
            cursor.execute("SELECT id FROM market_items WHERE url = ?", (item.url,))
            row = cursor.fetchone()
            return row[0] if row else -1

    def get_market_items(
        self,
        source: Optional[str] = None,
        is_artwork: Optional[bool] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[dict]:
        """Query market items with filters."""
        cursor = self.conn.cursor()

        query = "SELECT * FROM market_items WHERE 1=1"
        params = []

        if source:
            query += " AND source = ?"
            params.append(source)

        if is_artwork is not None:
            query += " AND is_artwork = ?"
            params.append(is_artwork)

        if min_price is not None:
            query += " AND price >= ?"
            params.append(min_price)

        if max_price is not None:
            query += " AND price <= ?"
            params.append(max_price)

        query += " ORDER BY timestamp_found DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        items = []
        for row in rows:
            item_dict = dict(row)
            # Parse analysis JSON if present
            if item_dict.get("analysis_json"):
                item_dict["analysis_json"] = json.loads(item_dict["analysis_json"])
            items.append(item_dict)

        return items

    def add_alert(self, alert: Alert) -> int:
        """Add an alert to the database."""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO alerts (item_id, alert_level, reason, notified)
            VALUES (?, ?, ?, ?)
        """, (alert.item_id, alert.alert_level, alert.reason, alert.notified))

        alert_id = cursor.lastrowid
        self.conn.commit()
        log.info(f"Created alert {alert_id} for item {alert.item_id}: {alert.reason}")
        return alert_id

    def get_alerts(self, notified: Optional[bool] = None, limit: int = 100) -> List[dict]:
        """Get alerts from the database."""
        cursor = self.conn.cursor()

        query = """
            SELECT a.*, m.title, m.url, m.price
            FROM alerts a
            JOIN market_items m ON a.item_id = m.id
            WHERE 1=1
        """
        params = []

        if notified is not None:
            query += " AND a.notified = ?"
            params.append(notified)

        query += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def mark_alert_notified(self, alert_id: int):
        """Mark an alert as notified."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE alerts SET notified = 1 WHERE id = ?", (alert_id,))
        self.conn.commit()

    def find_duplicate_by_hash(self, phash: str, threshold: int = 5) -> Optional[int]:
        """Find duplicate items by perceptual hash."""
        cursor = self.conn.cursor()

        # This is a simple exact match; for hamming distance, we'd need custom function
        cursor.execute("""
            SELECT item_id FROM market_images WHERE phash = ?
        """, (phash,))

        row = cursor.fetchone()
        return row[0] if row else None

    def close(self):
        """Close database connection."""
        self.conn.close()
        log.info("Database connection closed")


# Global database instance
db = Database()
