# Data Directory

This directory contains local data files, embeddings, databases, and sample data for the Gemäldeagent system.

## Directory Structure

- `embeddings/` - FAISS indices and embedding vectors for artist/style matching
- `historical_prices/` - Historical auction and market price data
- `mock_marketplace/` - Sample marketplace data for testing
- `sample_images/` - Sample artwork images for testing

## Files Not in Git

The following are generated at runtime and excluded from version control:
- `.db`, `.sqlite`, `.duckdb` - Database files
- `.faiss` - FAISS index files
- `.pkl`, `.npy` - Serialized embeddings
- Downloaded images from marketplace scans

## Initial Setup

Some sample data files will be created during first run or can be generated using:

```bash
python -m backend.core.setup_data
```

## Data Privacy

Do not commit:
- User-uploaded images
- Scraped marketplace data
- Generated reports
- Any personally identifiable information
