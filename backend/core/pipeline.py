"""
Main analysis pipeline that orchestrates all components.
"""
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from backend.analysis.artist_matching import artist_matcher
from backend.analysis.authenticity import authenticity_analyzer
from backend.analysis.condition import condition_analyzer
from backend.core.logging import log
from backend.core.models import AnalysisMetadata, AnalysisResult
from backend.provenance.provenance_helper import provenance_helper
from backend.valuation.valuation import valuation_engine
from backend.vision.pipeline import vision_pipeline


def analyze_artwork(
    image: Union[str, Path, Image.Image, bytes],
    notes: Optional[str] = None,
    source: str = "user_upload"
) -> AnalysisResult:
    """
    Complete artwork analysis pipeline.

    Orchestrates vision, analysis, valuation, and provenance components.

    Args:
        image: Image as file path, PIL Image, or bytes
        notes: Optional user notes
        source: Source identifier (e.g., "user_upload", "marketplace:ebay")

    Returns:
        Complete AnalysisResult
    """
    import time
    start_time = time.time()

    log.info("=" * 80)
    log.info("Starting complete artwork analysis")
    log.info("=" * 80)

    # Determine image path for reporting
    if isinstance(image, (str, Path)):
        image_path = str(image)
        pil_image = vision_pipeline.processor.load_image(image)
    elif isinstance(image, bytes):
        image_path = "uploaded_image"
        pil_image = vision_pipeline.processor.load_image_from_bytes(image)
    else:
        image_path = "pil_image"
        pil_image = image

    # Step 1: Vision analysis
    log.info("Step 1/6: Vision analysis")
    vision_result = vision_pipeline.process_image(pil_image)

    image_embedding = vision_result["embedding"]
    style_estimation = vision_result["style_estimation"]
    is_artwork = vision_result["is_artwork"]

    if not is_artwork:
        log.warning("Image does not appear to be an artwork!")

    # Step 2: Artist matching
    log.info("Step 2/6: Artist matching")
    artist_candidates = artist_matcher.match_artist(
        image_embedding,
        top_k=5,
        min_similarity=0.3
    )

    # Step 3: Condition analysis
    log.info("Step 3/6: Condition analysis")
    condition = condition_analyzer.analyze(pil_image)

    # Step 4: Authenticity assessment
    log.info("Step 4/6: Authenticity assessment")
    authenticity_result = authenticity_analyzer.analyze(
        image=pil_image,
        image_embedding=image_embedding,
        style_estimation=style_estimation,
        artist_candidates=artist_candidates,
        provenance_hints=None  # Will be filled after provenance research
    )

    # Step 5: Market valuation
    log.info("Step 5/6: Market valuation")
    valuation = valuation_engine.estimate_value(
        artist_candidates=artist_candidates,
        style=style_estimation.style,
        epoch=style_estimation.epoch,
        condition=condition,
        authenticity_score=authenticity_result.authenticity_score,
        size_category="medium"  # Could be determined from image
    )

    # Step 6: Provenance research
    log.info("Step 6/6: Provenance research")
    artist_name = artist_candidates[0].name if artist_candidates else None
    provenance = provenance_helper.research_provenance(
        artist_name=artist_name,
        title=None,
        style=style_estimation.style,
        epoch=style_estimation.epoch,
        image_path=image_path if isinstance(image, (str, Path)) else None
    )

    # Create metadata
    processing_time = time.time() - start_time
    metadata = AnalysisMetadata(
        source=source,
        notes=notes,
        processing_time_seconds=processing_time
    )

    # Assemble complete result
    result = AnalysisResult(
        image_path=image_path,
        style_estimation=style_estimation,
        artist_candidates=artist_candidates,
        authenticity_score=authenticity_result.authenticity_score,
        condition=condition,
        valuation=valuation,
        provenance=provenance,
        metadata=metadata
    )

    log.info("=" * 80)
    log.info(f"Analysis complete in {processing_time:.2f}s")
    log.info(f"  Style: {style_estimation.epoch} / {style_estimation.style}")
    log.info(f"  Top Artist: {artist_candidates[0].name if artist_candidates else 'Unknown'}")
    log.info(f"  Authenticity: {authenticity_result.authenticity_score}/100")
    log.info(f"  Estimated Value: €{valuation.estimated_value:,.0f}")
    log.info("=" * 80)

    return result
