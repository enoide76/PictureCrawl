"""
FastAPI application for Gemäldeagent.

Provides REST API endpoints for artwork analysis and market monitoring.
"""
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.core.config import settings
from backend.core.logging import log
from backend.core.models import (
    AnalyzeImageResponse,
    GenerateReportResponse,
    MarketItemsQuery,
    MarketItemsResponse,
    MarketScanResponse,
)
from backend.core.pipeline import analyze_artwork
from backend.pdf_reports.report_builder import report_builder

# Create FastAPI app
app = FastAPI(
    title="Gemäldeagent API",
    description="Intelligent Artwork Analysis System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    log.info("Starting Gemäldeagent API...")
    log.info(f"CORS origins: {settings.cors_origins_list}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    log.info("Shutting down Gemäldeagent API...")


# ============================================================================
# Health & Info Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Gemäldeagent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "analyze_image": "/api/analyze-image",
            "analyze_and_report": "/api/analyze-and-report",
            "market_items": "/api/market/items",
            "market_scan": "/api/monitoring/run-scan"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================================================
# Artwork Analysis Endpoints
# ============================================================================


@app.post("/api/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(
    image: UploadFile = File(..., description="Image file to analyze"),
    notes: str = Form(None, description="Optional notes about the artwork")
):
    """
    Analyze an artwork image.

    Returns complete analysis including style, artist candidates,
    authenticity, condition, and valuation.
    """
    try:
        log.info(f"Received analysis request for: {image.filename}")

        # Validate file type
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {image.content_type}"
            )

        # Read image bytes
        image_bytes = await image.read()

        # Perform analysis
        result = analyze_artwork(
            image=image_bytes,
            notes=notes,
            source="user_upload"
        )

        return AnalyzeImageResponse(
            success=True,
            result=result
        )

    except Exception as e:
        log.error(f"Error analyzing image: {e}", exc_info=True)
        return AnalyzeImageResponse(
            success=False,
            error=str(e)
        )


@app.post("/api/analyze-and-report")
async def analyze_and_report(
    image: UploadFile = File(...),
    notes: str = Form(None),
    format: str = Form("pdf", description="Report format: pdf or docx")
):
    """
    Analyze an artwork and generate a downloadable report.

    Returns analysis result and link to generated report.
    """
    try:
        log.info(f"Received analysis+report request for: {image.filename}")

        # Read image bytes
        image_bytes = await image.read()

        # Perform analysis
        result = analyze_artwork(
            image=image_bytes,
            notes=notes,
            source="user_upload"
        )

        # Generate report
        report_path = report_builder.generate_report(
            analysis_result=result,
            format=format,
            include_image=False  # Don't include uploaded bytes in report
        )

        return {
            "success": True,
            "analysis": result,
            "report_path": report_path,
            "download_url": f"/api/reports/download?path={report_path}"
        }

    except Exception as e:
        log.error(f"Error in analyze-and-report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/download")
async def download_report(path: str = Query(..., description="Report file path")):
    """Download a generated report file."""
    from pathlib import Path

    report_path = Path(path)

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf" if path.endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=report_path.name
    )


# ============================================================================
# Market Monitoring Endpoints
# ============================================================================


@app.get("/api/market/items", response_model=MarketItemsResponse)
async def get_market_items(
    source: str = Query(None, description="Filter by source (ebay, willhaben, etc.)"),
    is_artwork: bool = Query(None, description="Filter by artwork classification"),
    min_price: float = Query(None, description="Minimum price"),
    max_price: float = Query(None, description="Maximum price"),
    limit: int = Query(100, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset")
):
    """
    Query market items from monitoring database.

    Supports filtering by source, artwork status, and price range.
    """
    try:
        from backend.core.database import db

        items = db.get_market_items(
            source=source,
            is_artwork=is_artwork,
            min_price=min_price,
            max_price=max_price,
            limit=limit,
            offset=offset
        )

        # Get total count (simplified - in production would use separate count query)
        total = len(items)

        return MarketItemsResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        log.error(f"Error querying market items: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/monitoring/run-scan", response_model=MarketScanResponse)
async def run_market_scan(
    sources: list[str] = Query(None, description="Specific sources to scan"),
    max_items: int = Query(None, description="Max items to scan per source")
):
    """
    Manually trigger a market monitoring scan.

    Scans configured marketplaces for new artwork listings.
    """
    try:
        if not settings.MONITORING_ENABLED:
            raise HTTPException(
                status_code=403,
                detail="Market monitoring is disabled in configuration"
            )

        log.info("Manual market scan triggered")

        # Import monitoring module
        from backend.monitoring.scanner import run_monitoring_scan

        # Run scan
        results = run_monitoring_scan(
            sources=sources,
            max_items=max_items or settings.MONITORING_MAX_ITEMS_PER_SCAN
        )

        return MarketScanResponse(
            success=True,
            items_scanned=results.get("items_scanned", 0),
            artworks_found=results.get("artworks_found", 0),
            alerts_created=results.get("alerts_created", 0)
        )

    except Exception as e:
        log.error(f"Error running market scan: {e}", exc_info=True)
        return MarketScanResponse(
            success=False,
            error=str(e)
        )


@app.get("/api/monitoring/alerts")
async def get_alerts(
    notified: bool = Query(None, description="Filter by notification status"),
    limit: int = Query(100, description="Maximum results")
):
    """Get alerts from market monitoring."""
    try:
        from backend.core.database import db

        alerts = db.get_alerts(notified=notified, limit=limit)

        return {
            "success": True,
            "alerts": alerts,
            "count": len(alerts)
        }

    except Exception as e:
        log.error(f"Error getting alerts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Statistics & Analytics Endpoints
# ============================================================================


@app.get("/api/stats/overview")
async def get_stats_overview():
    """Get overview statistics."""
    try:
        from backend.core.database import db

        # Get counts
        all_items = db.get_market_items(limit=10000)
        artworks = [item for item in all_items if item.get("is_artwork")]
        alerts = db.get_alerts(limit=10000)

        return {
            "total_items": len(all_items),
            "total_artworks": len(artworks),
            "total_alerts": len(alerts),
            "unnotified_alerts": len([a for a in alerts if not a.get("notified")])
        }

    except Exception as e:
        log.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=settings.API_WORKERS
    )
