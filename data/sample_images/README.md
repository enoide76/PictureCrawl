# Sample Images Directory

This directory contains sample artwork images for testing and development.

## Purpose

- Testing the vision pipeline
- Validating analysis algorithms
- Demonstrating system capabilities
- Unit testing

## Structure

- Test images should be added here during development
- Images can be organized by style/epoch subdirectories if needed
- User uploads during runtime are stored separately

## Usage

```python
from backend.vision.pipeline import VisionPipeline

pipeline = VisionPipeline()
result = pipeline.process_image("data/sample_images/monet_sample.jpg")
```

## Note

Sample images are not included in the repository to keep it lightweight.
Users should add their own test images or download public domain artworks for testing.

## Public Domain Sources

- Wikimedia Commons
- Metropolitan Museum of Art API
- Rijksmuseum API
- Google Arts & Culture
