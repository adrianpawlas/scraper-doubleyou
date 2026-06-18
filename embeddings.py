"""
Embedding pipeline for the Double You Studios scraper.

Produces 768-dimensional L2-normalized embeddings using
google/siglip-base-patch16-384 (via local transformers) for:
- Product images (image_embedding / back_image_embedding)
- Product text metadata (info_embedding)

SigLIP is a vision-language model that provides both image and text
feature encoders in a shared embedding space. Both output 768-d vectors.

This approach matches the established pattern across all Finds scrapers:
all scrapers use local transformers with SigLIP, not the HF Inference API.
"""

import io
import logging
import time
from typing import Optional

import numpy as np
from PIL import Image

import torch
from transformers import AutoModel, AutoProcessor

from config import (
    EMBEDDING_DIM,
    EMBEDDING_VERSION,
    EMBEDDING_MODEL,
    HF_DELAY,
    IMAGE_JPEG_QUALITY,
    IMAGE_MAX_LONG_SIDE,
)

logger = logging.getLogger(__name__)


# ── Lazy-loaded model singleton ─────────────────────────────────────────────

_siglip_model: Optional[AutoModel] = None
_siglip_processor: Optional[AutoProcessor] = None


def _get_siglip():
    """Lazy-load the SigLIP model and processor (singleton)."""
    global _siglip_model, _siglip_processor
    if _siglip_model is None or _siglip_processor is None:
        logger.info("Loading SigLIP model: %s", EMBEDDING_MODEL)
        # Reset both to None before init so a partial failure doesn't leave
        # us in a broken state (model loaded, processor None).
        _siglip_model = None
        _siglip_processor = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _siglip_model = (
                AutoModel.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
                .to(device)
                .eval()
            )
            _siglip_processor = AutoProcessor.from_pretrained(
                EMBEDDING_MODEL, trust_remote_code=True
            )
            logger.info("SigLIP loaded on %s.", device)
        except Exception as exc:
            logger.error(
                "Failed to load SigLIP model/processor: %s", exc
            )
            _siglip_model = None
            _siglip_processor = None
            raise
    return _siglip_model, _siglip_processor


# ── Image pre-processing ────────────────────────────────────────────────────


def _prepare_image(image_data: bytes) -> Image.Image:
    """
    Process raw image bytes into a PIL Image ready for SigLIP.

    1. Decode to RGB.
    2. Resize longest side to max IMAGE_MAX_LONG_SIDE (preserve aspect ratio).
    3. Re-encode as JPEG quality ~85 for consistency.
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


# ── L2 normalization helper ─────────────────────────────────────────────────


def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding vector in-place."""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


# ── Image embedding ─────────────────────────────────────────────────────────


def embed_image(image_data: bytes) -> Optional[np.ndarray]:
    """
    Generate a 768-d L2-normalized embedding from product image bytes.

    Steps:
    1. Prepare image (RGB, resize max 1280px, JPEG re-encode).
    2. Run through SigLIP image encoder.
    3. L2-normalize.
    4. Validate shape and finiteness.
    """
    try:
        model, processor = _get_siglip()
        device = next(model.parameters()).device

        img = _prepare_image(image_data)

        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        embedding = outputs.cpu().numpy().flatten()
        embedding = _l2_normalize(embedding)

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


# ── Text embedding (SigLIP text encoder) ────────────────────────────────────


def embed_text(text: str) -> Optional[np.ndarray]:
    """
    Generate a 768-d L2-normalized embedding from product text metadata
    using the SigLIP text encoder.

    This matches the pattern used across all Finds scrapers — SigLIP
    is used for both image and text embeddings, keeping a single model
    dependency and producing aligned embeddings.

    Args:
        text: Concatenated text fields (title, description, category,
              gender, price, sale, metadata, tags, size).

    Returns:
        768-d L2-normalized numpy array, or None on failure.
    """
    try:
        if not text or not text.strip():
            logger.warning("Empty text for embedding.")
            return None

        model, processor = _get_siglip()
        device = next(model.parameters()).device

        # Truncate text to a reasonable max length to avoid OOM
        # SigLIP default max_length is 64, but we can allow up to 128
        max_len = min(len(text), 512)
        text = text[:max_len]

        inputs = processor(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.get_text_features(**inputs)

        embedding = outputs.cpu().numpy().flatten()
        embedding = _l2_normalize(embedding)

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

            meta_dict = (
                json.loads(metadata)
                if isinstance(metadata, str)
                else metadata
            )
            if isinstance(meta_dict, dict):
                for key in ["sku_list", "option_values"]:
                    val = meta_dict.get(key)
                    if val:
                        parts.append(str(val))
        except (json.JSONDecodeError, TypeError):
            parts.append(str(metadata))

    return " ".join(p for p in parts if p)
