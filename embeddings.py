"""
Embedding pipeline for the Double You Studios scraper.

Produces 768-dimensional L2-normalized embeddings for:
- Product images via google/siglip-base-patch16-384 (local transformers)
- Product text info via sentence-transformers/all-mpnet-base-v2 (local transformers)

Note on architecture:
The user specified using HuggingFace free Inference API for both models,
but google/siglip-base-patch16-384 is NOT available on the free serverless
Inference API (returns 503), and Gemini embedding-001 is a Google API that
is not hosted on HuggingFace at all. As a result, we run both models
locally using the transformers library, which is the most reliable approach
for CI/CD pipelines with model caching.
"""

import io
import logging
import time
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_DIM,
    EMBEDDING_VERSION,
    HF_DELAY,
    IMAGE_EMBEDDING_MODEL,
    IMAGE_JPEG_QUALITY,
    IMAGE_MAX_LONG_SIDE,
    TEXT_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


# ── Lazy-loaded model singletons ────────────────────────────────────────────

_siglip_model = None
_siglip_processor = None
_text_model = None


def _get_siglip():
    """Lazy-load the SigLIP model and processor."""
    global _siglip_model, _siglip_processor
    if _siglip_model is None:
        logger.info("Loading SigLIP model: %s", IMAGE_EMBEDDING_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _siglip_model = AutoModel.from_pretrained(
            IMAGE_EMBEDDING_MODEL, trust_remote_code=True
        ).to(device).eval()
        _siglip_processor = AutoProcessor.from_pretrained(
            IMAGE_EMBEDDING_MODEL, trust_remote_code=True
        )
        logger.info("SigLIP loaded on %s.", device)
    return _siglip_model, _siglip_processor


def _get_text_model():
    """Lazy-load the sentence-transformer text embedding model."""
    global _text_model
    if _text_model is None:
        logger.info("Loading text embedding model: %s", TEXT_EMBEDDING_MODEL)
        _text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        logger.info("Text embedding model loaded.")
    return _text_model


# ── Image pre-processing ────────────────────────────────────────────────────


def _prepare_image(image_data: bytes) -> Image.Image:
    """
    Process raw image bytes into a PIL Image ready for SigLIP.

    1. Decode to RGB.
    2. Resize longest side to max IMAGE_MAX_LONG_SIDE (preserve aspect ratio).
    3. Encode as JPEG quality ~85%.
    4. Decode again (to ensure consistent encoding).
    """
    img = Image.open(io.BytesIO(image_data))
    img = img.convert("RGB")

    # Resize: longest side
    w, h = img.size
    if max(w, h) > IMAGE_MAX_LONG_SIDE:
        scale = IMAGE_MAX_LONG_SIDE / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Re-encode as JPEG quality ~85 for consistency
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


# ── Image embedding (SigLIP) ────────────────────────────────────────────────


def embed_image(image_data: bytes) -> Optional[np.ndarray]:
    """
    Generate a 768-d L2-normalized embedding from product image bytes.

    Steps:
    1. Prepare image (RGB, resize max 1280px, JPEG re-encode)
    2. Run through SigLIP model
    3. L2-normalize the embedding
    4. Validate shape and finiteness
    """
    try:
        model, processor = _get_siglip()
        device = next(model.parameters()).device

        img = _prepare_image(image_data)

        # Process for SigLIP
        inputs = processor(
            images=img,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        embedding = outputs.cpu().numpy().flatten()

        # L2-normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Validate
        assert len(embedding) == EMBEDDING_DIM, (
            f"Expected {EMBEDDING_DIM} dims, got {len(embedding)}"
        )
        assert np.all(np.isfinite(embedding)), "Embedding contains NaN/Inf"

        time.sleep(HF_DELAY)
        return embedding

    except Exception as exc:
        logger.error("Image embedding failed: %s", exc)
        return None


# ── Text embedding (sentence-transformers) ──────────────────────────────────


def embed_text(text: str) -> Optional[np.ndarray]:
    """
    Generate a 768-d L2-normalized embedding from product text metadata.

    Args:
        text: Concatenated text fields (title + description + category +
              gender + price + sale + metadata + tags + size)

    Returns:
        768-d L2-normalized numpy array, or None on failure.
    """
    try:
        if not text or not text.strip():
            logger.warning("Empty text for embedding.")
            return None

        model = _get_text_model()
        embedding = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Validate
        assert len(embedding) == EMBEDDING_DIM, (
            f"Expected {EMBEDDING_DIM} dims, got {len(embedding)}"
        )
        assert np.all(np.isfinite(embedding)), "Text embedding contains NaN/Inf"

        time.sleep(HF_DELAY)
        return embedding

    except Exception as exc:
        logger.error("Text embedding failed: %s", exc)
        return None


# ── Convenience: build info text from product row ───────────────────────────


def build_info_text(row: dict) -> str:
    """
    Build a consolidated text string from product fields for info_embedding.
    """
    parts = [
        str(row.get("title", "")),
        str(row.get("description", "") or ""),
        str(row.get("category", "") or ""),
        str(row.get("gender", "") or ""),
        str(row.get("price", "") or ""),
        str(row.get("sale", "") or ""),
        str(row.get("size", "") or ""),
    ]
    tags = row.get("tags")
    if tags:
        if isinstance(tags, list):
            parts.append(" ".join(tags))
        else:
            parts.append(str(tags))
    metadata = row.get("metadata")
    if metadata:
        try:
            import json
            meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
            if isinstance(meta_dict, dict):
                # Extract useful fields
                for key in ["sku_list", "option_values"]:
                    val = meta_dict.get(key)
                    if val:
                        parts.append(str(val))
        except (json.JSONDecodeError, TypeError):
            parts.append(str(metadata))

    return " ".join(p for p in parts if p)
