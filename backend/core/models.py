"""
Pydantic models for data structures used throughout the system.
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Vision & Analysis Models
# ============================================================================


class StyleEstimation(BaseModel):
    """Style and epoch estimation result."""

    epoch: str = Field(..., description="Art epoch (e.g., Impressionism, Baroque)")
    style: str = Field(..., description="Specific art style")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ArtistCandidate(BaseModel):
    """Artist candidate with similarity score."""

    name: str = Field(..., description="Artist name")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    epoch: Optional[str] = Field(None, description="Artist's primary epoch")
    style: Optional[str] = Field(None, description="Artist's primary style")


class AuthenticityExplanations(BaseModel):
    """Detailed explanations for authenticity scoring."""

    style_match: str = Field(..., description="Style matching explanation")
    signature: str = Field(..., description="Signature analysis")
    texture: str = Field(..., description="Texture/brushstroke analysis")
    material: Optional[str] = Field(None, description="Material analysis")
    provenance: Optional[str] = Field(None, description="Provenance notes")


class AuthenticityResult(BaseModel):
    """Authenticity assessment result."""

    authenticity_score: int = Field(..., ge=0, le=100, description="Overall score 0-100")
    explanations: AuthenticityExplanations


class ConditionAnalysis(BaseModel):
    """Artwork condition analysis."""

    craquele: bool = Field(..., description="Craquelé pattern detected")
    yellowing: bool = Field(..., description="Yellowing detected")
    stains: bool = Field(..., description="Stains detected")
    cracks: List[str] = Field(default_factory=list, description="Crack locations")
    damage_score: float = Field(..., ge=0.0, le=1.0, description="Damage score (0=mint, 1=heavily damaged)")
    notes: str = Field(..., description="Additional condition notes")


class ValuationResult(BaseModel):
    """Market valuation result."""

    estimated_value: float = Field(..., ge=0, description="Estimated value in EUR")
    min: float = Field(..., ge=0, description="Minimum estimated value")
    max: float = Field(..., ge=0, description="Maximum estimated value")
    confidence: Literal["low", "medium", "high"] = Field(..., description="Confidence level")
    rationale: str = Field(..., description="Explanation of valuation")
    comparable_sales: int = Field(default=0, description="Number of comparable sales used")


class AuctionHistory(BaseModel):
    """Single auction history record."""

    date: str = Field(..., description="Sale date")
    auction_house: str = Field(..., description="Auction house name")
    price_realized: float = Field(..., description="Realized price")
    currency: str = Field(default="EUR", description="Currency")
    title: Optional[str] = Field(None, description="Lot title")


class ProvenanceResult(BaseModel):
    """Provenance research result."""

    reverse_image_hits: int = Field(..., ge=0, description="Number of reverse image search hits")
    auction_history: List[AuctionHistory] = Field(default_factory=list)
    notes: str = Field(..., description="Additional provenance notes")
    sources: List[str] = Field(default_factory=list, description="Information sources")


class AnalysisMetadata(BaseModel):
    """Metadata for analysis."""

    source: str = Field(..., description="Source of the image (user_upload, marketplace:name, etc.)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = Field(None, description="User-provided notes")
    processing_time_seconds: Optional[float] = Field(None, description="Total processing time")


class AnalysisResult(BaseModel):
    """Complete artwork analysis result - the standard format used throughout the system."""

    image_path: str = Field(..., description="Path to analyzed image")
    style_estimation: StyleEstimation
    artist_candidates: List[ArtistCandidate]
    authenticity_score: int = Field(..., ge=0, le=100)
    condition: ConditionAnalysis
    valuation: ValuationResult
    provenance: ProvenanceResult
    metadata: AnalysisMetadata


# ============================================================================
# Market Monitoring Models
# ============================================================================


class MarketItem(BaseModel):
    """Market item discovered during monitoring."""

    id: Optional[int] = None
    source: str = Field(..., description="Source platform (ebay, willhaben, etc.)")
    url: str = Field(..., description="Item URL")
    title: str = Field(..., description="Item title")
    description: Optional[str] = Field(None, description="Item description")
    price: float = Field(..., ge=0, description="Listed price")
    currency: str = Field(default="EUR")
    seller: Optional[str] = Field(None, description="Seller name/ID")
    location: Optional[str] = Field(None, description="Item location")
    timestamp_found: datetime = Field(default_factory=datetime.utcnow)
    image_urls: List[str] = Field(default_factory=list)
    image_hash: Optional[str] = Field(None, description="Perceptual hash of primary image")
    analysis_result: Optional[AnalysisResult] = Field(None, description="Full artwork analysis")
    is_artwork: bool = Field(default=False, description="Classified as artwork")


class Alert(BaseModel):
    """Alert for interesting market items."""

    id: Optional[int] = None
    item_id: int = Field(..., description="Associated market item ID")
    alert_level: Literal["info", "warning", "critical"] = Field(..., description="Alert severity")
    reason: str = Field(..., description="Reason for alert")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notified: bool = Field(default=False, description="Whether notification was sent")


# ============================================================================
# API Request/Response Models
# ============================================================================


class AnalyzeImageRequest(BaseModel):
    """Request for image analysis."""

    notes: Optional[str] = Field(None, description="Optional notes about the artwork")


class AnalyzeImageResponse(BaseModel):
    """Response for image analysis."""

    success: bool
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None


class GenerateReportRequest(BaseModel):
    """Request for report generation."""

    analysis_result: AnalysisResult
    output_format: Literal["pdf", "docx"] = Field(default="pdf")


class GenerateReportResponse(BaseModel):
    """Response for report generation."""

    success: bool
    report_path: Optional[str] = None
    error: Optional[str] = None


class MarketScanRequest(BaseModel):
    """Request to trigger market scan."""

    sources: Optional[List[str]] = Field(None, description="Specific sources to scan")
    max_items: Optional[int] = Field(None, description="Max items to scan")


class MarketScanResponse(BaseModel):
    """Response for market scan."""

    success: bool
    items_scanned: int = 0
    artworks_found: int = 0
    alerts_created: int = 0
    error: Optional[str] = None


class MarketItemsQuery(BaseModel):
    """Query parameters for market items."""

    source: Optional[str] = None
    is_artwork: Optional[bool] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class MarketItemsResponse(BaseModel):
    """Response for market items query."""

    items: List[MarketItem]
    total: int
    limit: int
    offset: int
