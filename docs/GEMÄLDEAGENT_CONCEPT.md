# Gemäldeagent - System Concept & Architecture

## Executive Summary

Gemäldeagent is an intelligent system that automatically:

- Recognizes paintings and artworks
- Analyzes artist, epoch, style, and material
- Assesses authenticity and quality indicators
- Generates market value and price estimates
- Enables provenance analysis
- Evaluates condition, restoration needs, and value development
- Integrates internet research (images, auctions, marketplaces)
- Produces user-friendly reports as PDF/DOCX

The system can run as a local tool, web service, or agent in a multi-agent system.

## 1. Goals

The Gemäldeagent aims to:

1. **Analyze artworks** from photographs
2. **Identify style and artist** using similarity matching and style classification
3. **Provide authenticity indicators** with transparent scoring
4. **Estimate market value** based on auction prices, trends, and comparable works
5. **Assess artwork condition** (damage, cracks, yellowing, craquelé)
6. **Recommend restoration** when appropriate
7. **Support provenance checks** via web search and databases
8. **Generate PDF reports** for collectors, insurance, and interested parties
9. **Monitor auction houses and marketplaces** (optional daily scanning)

## 2. System Architecture

### 2.1 Modular Structure

```
/gemäldeagent
 ├── backend/
 │    ├── core/              # Configuration, logging, database, shared models
 │    ├── vision/            # Image processing & embeddings
 │    ├── analysis/          # Artist matching, authenticity, condition
 │    ├── valuation/         # Market value estimation
 │    ├── provenance/        # Provenance research
 │    ├── pdf_reports/       # Report generation
 │    ├── monitoring/        # Market monitoring subsystem
 │    │    ├── marketplace_scanner/
 │    │    ├── auction_scanner/
 │    │    ├── flea_market_scanner/
 │    │    ├── image_extractor/
 │    │    ├── deduplication/
 │    │    └── alerting/
 │    └── api/               # FastAPI REST endpoints
 ├── frontend/               # User interface
 ├── data/                   # Local datasets
 └── docs/                   # Documentation
```

## 3. Core Components

### 3.1 Vision Layer (Image Processing)

**Functions:**
- Load and normalize images
- Color space analysis
- Style classification (CNN/ViT model)
- Feature extraction (brushstrokes, texture, craquelé)
- Object/motif recognition

**Models:**
- CLIP / SigLIP
- Vision Transformer (ViT)
- CNN Encoder
- OCR tools for signature detection

**Output:**
- Image embeddings (512-dim vector)
- Style classification (Impressionism, Baroque, etc.)
- Confidence scores
- Binary classification: is_artwork / not_artwork

### 3.2 Artist & Style Recognition

**Goals:**
- Match against reference database
- Similarity matching using embedding vector space
- Categorization by:
  - Epoch (Baroque, Renaissance, Impressionism, etc.)
  - Style (Expressionism, Cubism, etc.)
  - Artist-specific signatures (e.g., Van Gogh brushstroke patterns)

**Technical Implementation:**
- Embedding database (FAISS/DuckDB)
- Cosine similarity matching
- Threshold-based confidence scoring

**Output:**
```json
{
  "artist_candidates": [
    { "name": "Claude Monet", "similarity": 0.83 },
    { "name": "Camille Pissarro", "similarity": 0.74 }
  ],
  "style_estimation": {
    "epoch": "Impressionism",
    "style": "Landscape Impressionism",
    "confidence": 0.92
  }
}
```

### 3.3 Authenticity Indicator Module

**Evaluates:**
- Signature consistency
- Material & color palette (pigment analysis via color spectra)
- Brushstroke pattern comparison
- Craquelé patterns
- Provenance documentation (OCR + web search)
- Comparison with forgery patterns (optional GAN-fake detector)

**Output:**
- Score: 0-100
- Explanations per feature (transparent evaluation)

```json
{
  "authenticity_score": 78,
  "explanations": {
    "style_match": "High similarity to reference Monet works",
    "signature": "No clear signature detected",
    "texture": "Brushstroke pattern moderately consistent",
    "material": "Color palette consistent with period pigments"
  }
}
```

### 3.4 Condition & Damage Analysis

**Detected Defects:**
- Craquelé (age crackling)
- Yellowing (varnish aging)
- Moisture damage
- Cracks and tears
- Flaking paint
- Frame damage

**Techniques:**
- Edge detection
- Texture comparison
- Heatmap highlighting
- Pattern recognition for crack networks

**Output:**
```json
{
  "craquele": true,
  "yellowing": false,
  "stains": true,
  "cracks": ["lower_right"],
  "damage_score": 0.22,
  "notes": "Some visible surface cracks and stains in lower right quadrant."
}
```

### 3.5 Market Value Estimation

**Data Sources:**
- Online auction houses (Dorotheum, Sotheby's, Christie's)
- Marketplaces (eBay, Willhaben, Etsy)
- Historical price database
- Style-and-artist similarity weighted pricing

**Calculation:**
- Regression model (price prediction)
- Comparable work similarity
- Artist trend analysis
- Condition & authenticity score adjustments

**Output:**
```json
{
  "estimated_value": 45000,
  "min": 35000,
  "max": 70000,
  "confidence": "medium",
  "rationale": "Based on 12 similar sales of Monet-style impressionist landscapes."
}
```

### 3.6 Provenance Analysis

**Components:**
- Web search for:
  - Artist name
  - Motif/subject
  - Image title (if known or OCR-detected)
- Auction history of the artwork
- Hash comparison (reverse image search)

**Tools:**
- Local LLM for web result interpretation
- Google/Bing Custom Search API (optional)
- Reverse image tools (local or external)

**Output:**
```json
{
  "reverse_image_hits": 3,
  "auction_history": [
    {
      "date": "2019-05-12",
      "auction_house": "Dorotheum",
      "price_realized": 42000,
      "currency": "EUR"
    }
  ],
  "notes": "Found similar work sold in 2019"
}
```

### 3.7 PDF Report Generator

**Content:**
- Image of the artwork
- Artist/style analysis
- Authenticity score
- Condition & damage analysis
- Market value estimation
- Provenance notes
- Recommendations (restoration, insurance, sale)

**Technology:**
- ReportLab (PDF) / python-docx (DOCX)
- Configurable templates
- Multi-language support (planned)

## 4. Market Monitoring System

### 4.1 Goals

Automatically:
- Search websites
- Extract images
- Detect potential artworks/paintings
- Analyze relevant objects (style, artist, price, authenticity)
- Match against internal database
- Trigger alerts for worthwhile/rare works

**Target Users:**
- Art collectors
- Investors
- Galleries
- Private buyers
- Dealers

### 4.2 Architecture

```
/monitoring/
 ├── marketplace_scanner/    # eBay, Willhaben, Etsy scanners
 ├── auction_scanner/        # Auction house scrapers
 ├── flea_market_scanner/    # Classifieds platforms
 ├── image_extractor/        # Extract images from listings
 ├── deduplication/          # Perceptual hashing
 └── alerting/               # Alert rules & notifications
```

### 4.3 Core Functionalities

**1. Periodic Web Scraping**
- Every 30-120 minutes
- HTML scraping
- API scraping (where available)
- Browser automation (Playwright/Selenium)

**2. Image Extraction**
- From product listings
- From detail pages
- From social media posts
- From PDFs (auction catalogs)

**3. Classification: "Is it an artwork?"**
- Vision model decides:
  - Painting or not
  - Hand-painted or print
  - Style detection
  - Authenticity probability
  - Material (oil/canvas, watercolor, lithograph)

**4. Storage in Market Database**
```json
{
  "source": "ebay",
  "url": "https://....",
  "title": "Antique Oil Painting Farmhouse",
  "price": 450,
  "currency": "EUR",
  "image_urls": ["..."],
  "analysis": {
    "style": "Impressionism",
    "is_artwork": true,
    "artist_candidates": ["Pissarro", "Monet"],
    "value_estimate": {
      "min": 4000,
      "max": 8000
    }
  }
}
```

### 4.4 Agents

**MarketCrawlerAgent**
- Main web scraping agent
- Respects robots.txt
- Regional filtering
- Rate limiting

**ImageDetectorAgent**
- Extracts images
- Classifies: Painting/Drawing/Graphic/Not relevant

**ArtworkAnalyzerAgent**
- Runs full analysis pipeline on detected artworks
- Stores results in database

**MarketValuationAgent**
- Aggregates price data
- Tracks trends
- Detects bargains/mispricings

**AlertAgent**
- Evaluates alert rules:
  - Market price < 40% of estimated value
  - High artist similarity
  - Rare epoch/style
  - Unusual motifs
  - Good condition + high value
- Sends notifications:
  - Email
  - Telegram
  - Web dashboard

### 4.5 Platform List

**MVP:**
- eBay
- Willhaben
- Etsy
- Dorotheum
- Facebook Marketplace

**Extensions:**
- Christie's
- Sotheby's
- Artnet
- Artprice
- Instagram hashtags

### 4.6 Deduplication Engine

**Algorithm:**
- Perceptual hash (pHash, dHash)
- Duplicate detection: similarity > 0.95
- Title/price heuristics

**Goal:** No duplicate entries in system

### 4.7 Database Schema

**market_items**
- id, source, url, title, description
- price, currency, timestamp_found
- image_hash, analysis_json

**market_images**
- id, item_id, image_url
- phash, embedding_vector

**alerts**
- id, item_id, alert_level
- reason, created_at

## 5. Data Structures

### 5.1 Standard Analysis Result

All components share this common format:

```json
{
  "image_path": "input.jpg",
  "style_estimation": {
    "epoch": "Impressionism",
    "style": "Landscape Impressionism",
    "confidence": 0.92
  },
  "artist_candidates": [
    { "name": "Claude Monet", "similarity": 0.83 },
    { "name": "Camille Pissarro", "similarity": 0.74 }
  ],
  "authenticity_score": 78,
  "condition": {
    "craquele": true,
    "yellowing": false,
    "stains": true,
    "damage_score": 0.22
  },
  "valuation": {
    "estimated_value": 45000,
    "min": 35000,
    "max": 70000,
    "confidence": "medium"
  },
  "provenance": {
    "reverse_image_hits": 3,
    "auction_history": [],
    "notes": "Found similar work sold in 2019"
  },
  "metadata": {
    "source": "user_upload",
    "created_at": "2024-01-15T10:30:00Z",
    "notes": "User suspected: Monet"
  }
}
```

## 6. Technology Stack

**Vision / ML:**
- PyTorch
- Transformers (ViT, CLIP)
- OpenCV
- Scikit-image

**Backend:**
- FastAPI
- Python 3.11
- DuckDB (embedding DB)
- FAISS (similarity search)

**Frontend:**
- React / Next.js
- Upload UI + results view

**Web Scraping:**
- BeautifulSoup4
- Playwright
- Selenium
- Requests

**Reports:**
- ReportLab / python-docx

**Optional:**
- LangGraph (agent orchestration)
- Local LLM (Ollama / Llama3)

## 7. End-to-End Workflow

1. User uploads image
2. VisionAgent extracts features + style
3. ArtistMatcherAgent searches similar artists
4. AuthenticityAgent evaluates authenticity
5. ConditionAgent detects damage
6. ValuationAgent estimates price
7. ProvenanceAgent checks web sources
8. ReportAgent builds PDF
9. Output to user

## 8. Future Extensions

- Mobile app (photo on-site)
- AR comparison with museum databases
- Live auction monitoring
- Collector portfolio tracking
- Integration with insurance systems
- RAG art knowledge base
- Multi-region scanning (EU/USA/Asia)
- Deep learning style matching optimization
- Price history tracking
- Anomaly detection (fake offers)
- Seller/dealer history analysis

## 9. Security & Governance

- Local image processing (no mandatory cloud upload)
- Watermark detection
- Bias checks in valuations
- Transparent scoring explanations
- Logging & audit trails for expert review
- Respect robots.txt and rate limits
- GDPR compliance for stored data

## 10. Implementation Roadmap

**Phase 0:** Repository initialization ✓
**Phase 1:** Core image analysis (vision pipeline)
**Phase 2:** Artist matching & authenticity
**Phase 3:** Condition & valuation
**Phase 4:** Provenance & report generation
**Phase 5:** FastAPI backend
**Phase 6:** Basic frontend
**Phase 7:** Monitoring subsystem
**Phase 8:** Documentation & polish

---

*This document serves as the architectural blueprint for the Gemäldeagent system.*
