# Embeddings Directory

This directory stores vector embeddings and FAISS indices for artist/style matching.

## Files

- `artist_embeddings.faiss` - FAISS index of reference artwork embeddings
- `artist_metadata.json` - Metadata for reference artworks (artist names, styles, epochs)
- `style_embeddings.pkl` - Pickled embeddings organized by style
- `embedding_config.json` - Configuration for embedding generation

## Generation

Embeddings are generated during system initialization or can be regenerated using:

```bash
python -m backend.vision.generate_embeddings
```

## Format

Embeddings are stored as:
- FAISS indices for fast similarity search
- JSON metadata for human-readable information
- Pickle files for numpy arrays

## Not in Git

All `.faiss`, `.pkl`, and `.npy` files are excluded from version control due to size.
