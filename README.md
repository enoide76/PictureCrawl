# Gemäldeagent - Intelligent Artwork Analysis System

An intelligent system for analyzing paintings and artworks, providing style detection, artist identification, authenticity assessment, condition analysis, market valuation, and automated marketplace monitoring.

## Overview

Gemäldeagent is a comprehensive artwork analysis platform that combines computer vision, machine learning, and web scraping to:

- **Analyze artworks** from uploaded images
- **Detect style, epoch, and artist** using deep learning embeddings
- **Compute authenticity indicators** with transparent scoring
- **Analyze condition** (craquelé, damage, yellowing, etc.)
- **Estimate market value** based on comparable works
- **Research provenance** using web sources
- **Generate professional PDF/DOCX reports**
- **Monitor online marketplaces** for potentially valuable artworks

## Project Structure

```
gemäldeagent/
├── backend/                    # Python backend
│   ├── core/                  # Shared utilities, config, logging, db, models
│   ├── vision/                # Image processing, embeddings, style detection
│   ├── analysis/              # Artist/style matching, authenticity, condition
│   ├── valuation/             # Price estimation and market value logic
│   ├── provenance/            # Web lookup / provenance helper logic
│   ├── monitoring/            # Market monitoring & scraping subsystem
│   │   ├── marketplace_scanner/
│   │   ├── auction_scanner/
│   │   ├── flea_market_scanner/
│   │   ├── image_extractor/
│   │   ├── deduplication/
│   │   └── alerting/
│   ├── pdf_reports/           # PDF/DOCX report generation
│   ├── api/                   # FastAPI endpoints
│   └── tests/                 # Pytest tests
├── frontend/                   # Simple UI (React or HTML+JS)
├── data/                      # Local data (embeddings, DBs, configs)
│   ├── embeddings/
│   ├── historical_prices/
│   ├── mock_marketplace/
│   └── sample_images/
├── docs/                      # Documentation
│   ├── GEMÄLDEAGENT_CONCEPT.md
│   ├── API_REFERENCE.md
│   └── MONITORING_DESIGN.md
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

### Core Analysis
- **Vision Pipeline**: Image preprocessing, embedding generation (CLIP/ViT), style classification
- **Artist Matching**: Similarity search using FAISS embeddings against reference database
- **Authenticity Scoring**: 0-100 score with detailed explanations
- **Condition Analysis**: Automatic detection of craquelé, yellowing, stains, cracks, damage
- **Market Valuation**: Price estimation with confidence intervals
- **Provenance Research**: Web-based provenance lookup (optional)

### Market Monitoring
- **Automated Scanning**: Monitor eBay, Willhaben, Etsy, auction sites, flea markets
- **Image Detection**: Identify potential artworks automatically
- **Deduplication**: Perceptual hashing to avoid duplicate entries
- **Alert System**: Notifications for underpriced or rare pieces
- **Market Database**: Store and track discovered artworks

### Reporting
- **PDF Reports**: Professional reports for insurance, collectors, private use
- **DOCX Export**: Editable Word documents
- **Comprehensive Data**: Includes all analysis results, images, and metadata

## Installation

### Prerequisites
- Python 3.11+
- pip or conda
- (Optional) CUDA-capable GPU for faster inference

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd gemäldeagent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Playwright browsers (for web scraping)**
```bash
playwright install
```

5. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Running the Backend API

```bash
# From project root
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation (auto-generated): `http://localhost:8000/docs`

### Running the Frontend

```bash
cd frontend
# Instructions will be added in Phase 6
```

### Analyzing an Artwork

**Via API:**
```bash
curl -X POST "http://localhost:8000/api/analyze-image" \
  -F "image=@path/to/artwork.jpg" \
  -F "notes=Optional notes about the artwork"
```

**Via Python:**
```python
from backend.core.pipeline import analyze_artwork

result = analyze_artwork("path/to/artwork.jpg", notes="Suspected Monet")
print(result)
```

### Running Market Monitoring

**Manual scan:**
```bash
python -m backend.monitoring.run_scan
```

**Via API:**
```bash
curl -X POST "http://localhost:8000/api/monitoring/run-scan"
```

### Generating Reports

```python
from backend.pdf_reports.report_builder import generate_report

report_path = generate_report(
    analysis_result=result,
    output_path="reports/artwork_analysis.pdf"
)
```

## Configuration

### Environment Variables (.env)

```env
# Database
DATABASE_URL=sqlite:///data/gemäldeagent.db

# Vision Models
VISION_MODEL=openai/clip-vit-base-patch32
EMBEDDING_DIM=512

# API Keys (optional)
GOOGLE_CUSTOM_SEARCH_API_KEY=
GOOGLE_CUSTOM_SEARCH_ENGINE_ID=

# Monitoring
MONITORING_ENABLED=true
MONITORING_INTERVAL_MINUTES=60

# Alerts
ALERT_EMAIL_ENABLED=false
ALERT_EMAIL_TO=
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=

# Report Settings
REPORT_FORMAT=pdf  # pdf or docx
REPORT_OUTPUT_DIR=reports/
```

## Development

### Running Tests

```bash
pytest backend/tests/ -v --cov=backend
```

### Code Structure

- **backend/core**: Shared configuration, database models, logging
- **backend/vision**: Computer vision pipeline
- **backend/analysis**: Artist matching, authenticity, condition analysis
- **backend/valuation**: Market value estimation
- **backend/monitoring**: Web scraping and market monitoring
- **backend/api**: FastAPI REST endpoints

### Adding New Marketplaces

1. Create a new scanner in `backend/monitoring/marketplace_scanner/`
2. Inherit from `BaseMarketplaceScanner`
3. Implement `scan()` method
4. Register in monitoring configuration

## Technology Stack

- **Backend**: FastAPI, Python 3.11
- **Vision**: PyTorch, Transformers (CLIP/ViT), OpenCV, scikit-image
- **Database**: DuckDB, SQLite
- **Search**: FAISS (similarity search)
- **Web Scraping**: BeautifulSoup4, Playwright, Selenium
- **Reports**: ReportLab (PDF), python-docx (DOCX)
- **Testing**: pytest

## Roadmap

- [x] Phase 0: Repository initialization
- [ ] Phase 1: Core image analysis (vision pipeline)
- [ ] Phase 2: Artist matching & authenticity
- [ ] Phase 3: Condition & valuation
- [ ] Phase 4: Provenance & report generation
- [ ] Phase 5: FastAPI backend
- [ ] Phase 6: Basic frontend
- [ ] Phase 7: Monitoring subsystem
- [ ] Phase 8: Documentation & polish

## Documentation

- [Concept Document](docs/GEMÄLDEAGENT_CONCEPT.md) - Detailed system design
- [API Reference](docs/API_REFERENCE.md) - REST API documentation (coming soon)
- [Monitoring Design](docs/MONITORING_DESIGN.md) - Market monitoring architecture (coming soon)

## License

[Specify license]

## Contributing

[Specify contribution guidelines]

## Support

For issues and questions, please open an issue on GitHub.

---

**Note**: This is an MVP implementation. Some features use simplified heuristics that can be replaced with more sophisticated ML models as needed.
