"""
Configuration module for the Double You Studios product scraper.

All environment-specific settings are loaded from environment variables
or a .env file. No secrets are hard-coded.
"""

import os
from typing import Optional


# ── Brand / source ──────────────────────────────────────────────────────────
BRAND: str = "Double You Studios"
SOURCE: str = "scraper-doubleyou"
SECOND_HAND: bool = False

# ── Store URLs ──────────────────────────────────────────────────────────────
LANDING_PAGE: str = "https://doubleyou-studios.com"
CATEGORY_URLS: list[str] = [
    "https://doubleyou-studios.com/collections/all",
]

# ── Rate limiting (seconds between requests) ────────────────────────────────
REQUEST_DELAY: float = float(os.environ.get("SCRAPER_REQUEST_DELAY", "1.0"))
HF_DELAY: float = float(os.environ.get("SCRAPER_HF_DELAY", "0.5"))
STORE_REQUEST_DELAY: float = float(
    os.environ.get("SCRAPER_STORE_REQUEST_DELAY", "1.5")
)

# ── Batching ────────────────────────────────────────────────────────────────
BATCH_SIZE: int = int(os.environ.get("SCRAPER_BATCH_SIZE", "50"))

# ── Supabase ────────────────────────────────────────────────────────────────
SUPABASE_URL: Optional[str] = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.environ.get("SUPABASE_KEY")

# ── HuggingFace (optional for local model fallback) ─────────────────────────
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN", None)

# ── Embedding models ────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "google/siglip-base-patch16-384"
EMBEDDING_VERSION: int = 2
EMBEDDING_DIM: int = 768

# ── Image processing ────────────────────────────────────────────────────────
IMAGE_MAX_LONG_SIDE: int = 1280
IMAGE_JPEG_QUALITY: int = 85
