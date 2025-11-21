"""
Configuration management for Gemäldeagent.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    HISTORICAL_PRICES_DIR: Path = DATA_DIR / "historical_prices"
    MOCK_MARKETPLACE_DIR: Path = DATA_DIR / "mock_marketplace"
    SAMPLE_IMAGES_DIR: Path = DATA_DIR / "sample_images"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"

    # Database
    DATABASE_URL: str = Field(default="sqlite:///data/gemäldeagent.db")
    DUCKDB_PATH: str = Field(default="data/embeddings.duckdb")

    # Vision & ML Models
    VISION_MODEL: str = Field(default="openai/clip-vit-base-patch32")
    EMBEDDING_DIM: int = Field(default=512)
    DEVICE: Literal["cuda", "mps", "cpu"] = Field(default="cpu")
    MODEL_CACHE_DIR: str = Field(default="models/")

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_RELOAD: bool = Field(default=True)
    API_WORKERS: int = Field(default=1)
    CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8000")

    # External APIs (Optional)
    GOOGLE_CUSTOM_SEARCH_API_KEY: Optional[str] = Field(default=None)
    GOOGLE_CUSTOM_SEARCH_ENGINE_ID: Optional[str] = Field(default=None)

    # Market Monitoring
    MONITORING_ENABLED: bool = Field(default=True)
    MONITORING_INTERVAL_MINUTES: int = Field(default=60)
    MONITORING_MAX_ITEMS_PER_SCAN: int = Field(default=100)

    # Marketplace APIs
    EBAY_API_KEY: Optional[str] = Field(default=None)
    EBAY_APP_ID: Optional[str] = Field(default=None)
    WILLHABEN_ENABLED: bool = Field(default=True)
    ETSY_API_KEY: Optional[str] = Field(default=None)

    # Scraping behavior
    SCRAPING_DELAY_SECONDS: float = Field(default=2.0)
    SCRAPING_MAX_RETRIES: int = Field(default=3)
    SCRAPING_TIMEOUT_SECONDS: int = Field(default=30)
    USER_AGENT: str = Field(
        default="Mozilla/5.0 (compatible; GemäldeagentBot/1.0)"
    )

    # Deduplication
    DEDUPLICATION_ENABLED: bool = Field(default=True)
    PERCEPTUAL_HASH_THRESHOLD: int = Field(default=5)

    # Alert System
    ALERT_ENABLED: bool = Field(default=True)
    ALERT_BARGAIN_THRESHOLD: float = Field(default=0.4)
    ALERT_MIN_AUTHENTICITY_SCORE: int = Field(default=60)
    ALERT_RARE_STYLES: str = Field(
        default="Impressionism,Post-Impressionism,Expressionism"
    )

    # Email alerts
    ALERT_EMAIL_ENABLED: bool = Field(default=False)
    ALERT_EMAIL_TO: Optional[str] = Field(default=None)
    SMTP_HOST: Optional[str] = Field(default=None)
    SMTP_PORT: int = Field(default=587)
    SMTP_USER: Optional[str] = Field(default=None)
    SMTP_PASSWORD: Optional[str] = Field(default=None)
    SMTP_FROM: str = Field(default="gemäldeagent@example.com")

    # Telegram alerts
    ALERT_TELEGRAM_ENABLED: bool = Field(default=False)
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None)
    TELEGRAM_CHAT_ID: Optional[str] = Field(default=None)

    # Report Generation
    REPORT_FORMAT: Literal["pdf", "docx"] = Field(default="pdf")
    REPORT_OUTPUT_DIR: str = Field(default="reports/")
    REPORT_TEMPLATE: str = Field(default="default")
    REPORT_INCLUDE_IMAGES: bool = Field(default=True)
    REPORT_LANGUAGE: Literal["en", "de"] = Field(default="en")

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    LOG_FILE: str = Field(default="logs/gemäldeagent.log")
    LOG_FORMAT: Literal["json", "text"] = Field(default="text")

    # Frontend
    FRONTEND_URL: str = Field(default="http://localhost:3000")

    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION_MINUTES: int = Field(default=60)

    # Upload limits
    MAX_UPLOAD_SIZE_MB: int = Field(default=10)
    ALLOWED_EXTENSIONS: str = Field(default="jpg,jpeg,png,webp,bmp")

    # Development
    DEBUG: bool = Field(default=False)
    TESTING: bool = Field(default=False)
    MOCK_MARKETPLACES: bool = Field(default=True)

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def allowed_extensions_list(self) -> list[str]:
        """Get allowed file extensions as a list."""
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]

    @property
    def alert_rare_styles_list(self) -> list[str]:
        """Get rare styles as a list."""
        return [style.strip() for style in self.ALERT_RARE_STYLES.split(",")]

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.EMBEDDINGS_DIR,
            self.REPORTS_DIR,
            Path(self.LOG_FILE).parent,
            Path(self.MODEL_CACHE_DIR),
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
