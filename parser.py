"""
Parser module for the Double You Studios Shopify store.

Extracts product data from Shopify JSON endpoints. The store exposes:
- Collection products: /collections/all/products.json?page=N
- Single product:    /products/{handle}.json

Back-view detection rule (documented in README):
Images are scanned for filenames containing "Back" (case-insensitive)
while excluding size-guide images. The first non-guide image with "Back"
in its filename is treated as the back view.
"""

import hashlib
import json
import logging
import re
import time
from typing import Any, Optional

import httpx

from config import (
    BRAND,
    CATEGORY_URLS,
    SECOND_HAND,
    SOURCE,
    STORE_REQUEST_DELAY,
)

logger = logging.getLogger(__name__)


# ── HTTP helpers ────────────────────────────────────────────────────────────


def _build_client() -> httpx.Client:
    """Create an HTTPX client with sensible defaults."""
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=15.0),
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; DoubleYouScraper/1.0; "
                "+https://github.com/adrianpawlas/scraper-doubleyou)"
            ),
            "Accept": "application/json",
        },
    )


# ── Product URL discovery ───────────────────────────────────────────────────


def discover_all_product_urls(max_retries: int = 3) -> list[str]:
    """
    Iterate over all collection pages until an empty page is returned.

    Pagination: /collections/all/products.json?page=1, ?page=2, …
    When a page with 0 products is returned, scraping stops.
    Retries on 5xx errors with exponential backoff.
    """
    urls: list[str] = []

    for category_url in CATEGORY_URLS:
        base_url = category_url.rstrip("/") + "/products.json"
        page = 1
        while True:
            paginated_url = f"{base_url}?page={page}"
            logger.info("Fetching product list: %s", paginated_url)

            success = False
            pagination_done = False
            for attempt in range(1, max_retries + 1):
                with _build_client() as client:
                    try:
                        resp = client.get(paginated_url)
                        resp.raise_for_status()
                        data = resp.json()
                        products = data.get("products", [])

                        if not products:
                            logger.info(
                                "Empty page %d — done paginating.", page
                            )
                            success = True
                            pagination_done = True
                            break

                        for product in products:
                            handle = product.get("handle")
                            if handle:
                                product_url = (
                                    f"https://doubleyou-studios.com"
                                    f"/products/{handle}"
                                )
                                urls.append(product_url)
                        page += 1
                        success = True
                        break
                    except httpx.HTTPStatusError as exc:
                        if (
                            exc.response.status_code >= 500
                            and attempt < max_retries
                        ):
                            backoff = 2 ** attempt
                            logger.warning(
                                "Server error %d on %s, retrying in "
                                "%ds (attempt %d/%d)",
                                exc.response.status_code, paginated_url,
                                backoff, attempt, max_retries,
                            )
                            time.sleep(backoff)
                            continue
                        logger.error(
                            "Failed to fetch %s: %s", paginated_url, exc
                        )
                        break
                    except Exception as exc:
                        logger.error(
                            "Failed to fetch %s: %s", paginated_url, exc
                        )
                        break

            if not success:
                break
            if pagination_done:
                break

            time.sleep(STORE_REQUEST_DELAY)

    logger.info("Discovered %d product URLs.", len(urls))
    return urls


# ── Single product fetch ────────────────────────────────────────────────────


def fetch_product_json(
    product_url: str, max_retries: int = 3
) -> Optional[dict]:
    """
    Fetch the full product JSON from Shopify's /products/{handle}.json.
    Retries on 5xx errors with exponential backoff.
    Returns the inner 'product' dict, or None on failure.
    """
    json_url = product_url.rstrip("/") + ".json"
    for attempt in range(1, max_retries + 1):
        with _build_client() as client:
            try:
                resp = client.get(json_url)
                resp.raise_for_status()
                data = resp.json()
                return data.get("product")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < max_retries:
                    backoff = 2 ** attempt
                    logger.warning(
                        "Server error %d on %s, retrying in %ds "
                        "(attempt %d/%d)",
                        exc.response.status_code, json_url,
                        backoff, attempt, max_retries,
                    )
                    time.sleep(backoff)
                    continue
                logger.error("Failed to fetch %s: %s", json_url, exc)
                return None
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", json_url, exc)
                return None
    return None


# ── Back-view detection patterns (compiled once) ───────────────────────────

_back_pattern = re.compile(r'(?:^|[\W_])(back)(?:[\W_]|$)', re.IGNORECASE)
_rear_pattern = re.compile(r'(?:^|[\W_])(rear)(?:[\W_]|$)', re.IGNORECASE)
_b_suffix_pattern = re.compile(r'[\W_]b[\.]', re.IGNORECASE)


# ── Image classification ────────────────────────────────────────────────────


def _is_size_guide(filename: str) -> bool:
    """Return True if the filename appears to be a size guide / size chart."""
    lower = filename.lower()
    return any(
        kw in lower
        for kw in ["sizeguide", "sizechart", "sizeguide", "size guide",
                    "size_chart", "size-chart"]
    )


def _is_back_view(filename: str, alt: str = "") -> bool:
    """
    Detect if an image filename/alt text indicates a back view.

    Matches patterns (case-insensitive):
    - "back" as a word (bounded by _, -, ., start/end of string)
    - "rear" as a word
    - "_b." or "-b." (common convention for back views)
    """
    name_lower = filename.lower()
    alt_lower = alt.lower()

    # Pattern 1: "back" as a word bounded by non-alphanumeric chars
    if _back_pattern.search(name_lower) or _back_pattern.search(alt_lower):
        return True

    # Pattern 2: "rear" as a word
    if _rear_pattern.search(name_lower) or _rear_pattern.search(alt_lower):
        return True

    # Pattern 3: "_b." or "-b." (common convention for back view variant)
    if _b_suffix_pattern.search(name_lower):
        return True

    return False


def _classify_product_images(
    images: list[dict],
) -> tuple[Optional[str], Optional[str], list[str]]:
    """
    Classify product images into front, back, and gallery.

    Returns (front_image_url, back_image_url, additional_image_urls).

    Rules:
    - Front: position == 1, not a size guide, and NOT a back image.
    - Back: any image (any position) with back-view indicator in its
            filename (case-insensitive), excluding size guides.
            Detection patterns: "back", "rear", "_b." (as word boundaries)
    - Gallery: all non-front, non-back, non-size-guide images.
    - Back images are also included in additional_images.

    Some products (e.g. Studio Sport Tees) have position 1 as the BACK
    and position 2 as the FRONT. We handle this by checking filenames.
    """
    front_url: Optional[str] = None
    back_url: Optional[str] = None
    gallery: list[str] = []

    # First pass: identify back images and size guides
    back_candidates: list[str] = []
    size_guide_indices: set[int] = set()

    for i, img in enumerate(images):
        src = img.get("src", "")
        filename = src.rstrip("/").split("/")[-1].split("?")[0]
        alt = (img.get("alt") or "").lower()

        if _is_size_guide(filename) or _is_size_guide(alt):
            size_guide_indices.add(i)
            continue

        # Check for back view using word-boundary patterns
        if _is_back_view(filename, alt):
            back_candidates.append(src)
            continue

    # Second pass: determine front
    # Front is position 1 (first image) UNLESS it's a back image
    # If position 1 is a back image, use position 2 as front
    for i, img in enumerate(images):
        if i in size_guide_indices:
            continue
        src = img.get("src", "")
        if src in back_candidates:
            continue
        if front_url is None:
            front_url = src
            break

    # If all non-guide images are back views, use the first image anyway
    if front_url is None and images:
        for i, img in enumerate(images):
            if i not in size_guide_indices:
                front_url = img.get("src")
                break
        # Last resort: use position 0
        if front_url is None:
            front_url = images[0].get("src")

    # Set back from first candidate
    if back_candidates:
        back_url = back_candidates[0]

    # Build gallery: non-front, non-back, non-size-guide
    for i, img in enumerate(images):
        if i in size_guide_indices:
            continue
        src = img.get("src", "")
        if src == front_url:
            continue
        if src == back_url:
            continue
        gallery.append(src)

    # Also include back in gallery
    if back_url and back_url not in gallery:
        gallery.append(back_url)

    return front_url, back_url, gallery


# ── Price parsing ───────────────────────────────────────────────────────────


def _parse_price(
    variants: list[dict],
) -> str:
    """
    Return a price string from the first available variant.
    Format: "1681.29CZK"
    Shopify provides variant-level prices and currency.
    """
    if not variants:
        return ""
    v = variants[0]
    price = v.get("price", "0.00")
    currency = v.get("price_currency", "CZK")
    return f"{price}{currency}"


def _parse_sale_price(variants: list[dict]) -> Optional[str]:
    """
    Return the sale price (compare_at_price) if any variant has one set,
    otherwise None.
    Format: "1500.00CZK"
    """
    for v in variants:
        compare = v.get("compare_at_price")
        if compare:
            currency = v.get("price_currency", "CZK")
            return f"{compare}{currency}"
    return None


# ── Category derivation ─────────────────────────────────────────────────────


def _derive_category(title: str, product_type: str) -> Optional[str]:
    """Derive a product category from the title or product_type."""
    if product_type:
        return product_type

    title_lower = title.lower()

    # Map of keywords → categories
    category_map: list[tuple[list[str], str]] = [
        (["jacket", "bomber", "workwear", "spiral", "reversible", "grid"],
         "Jackets"),
        (["knit", "sweater", "jumper", "cardigan", "brushed"], "Knitwear"),
        (["polo"], "Polos"),
        (["shirt"], "Shirts"),
        (["t-shirt", "tee", "t shirt"], "T-Shirts"),
        (["pant", "trouser", "jean", "denim", "shorts", "cargo"], "Bottoms"),
        (["beanie", "cap", "hat"], "Headwear"),
        (["hoodie", "sweatshirt", "fleece"], "Hoodies"),
    ]

    matched: list[str] = []
    for keywords, category in category_map:
        if any(kw in title_lower for kw in keywords):
            matched.append(category)

    return ", ".join(matched) if matched else None


# ── ID generation ───────────────────────────────────────────────────────────


def generate_product_id(source: str, product_url: str) -> str:
    """Stable product ID from source + product_url hash (UUID format)."""
    raw = f"{source}:{product_url}"
    hex_digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    # Format as UUID: 8-4-4-4-12 (compatible with uuid column type)
    return f"{hex_digest[:8]}-{hex_digest[8:12]}-{hex_digest[12:16]}-{hex_digest[16:20]}-{hex_digest[20:32]}"


# ── Metadata builder ────────────────────────────────────────────────────────


def build_metadata(
    product: dict,
    variants: list[dict],
    scrape_timestamp: str,
) -> str:
    """
    Build a JSON metadata string with all extra product info.

    Includes: sizes, colors, composition, care instructions, SKU,
    availability, weight, scrape timestamp, etc.
    """
    options = product.get("options", [])
    option_names = [o.get("name", "") for o in options]
    option_values = {
        o.get("name", ""): o.get("values", []) for o in options
    }

    variant_details = []
    all_skus = []
    all_available = False
    for v in variants:
        all_skus.append(v.get("sku", ""))
        if v.get("available"):
            all_available = True
        variant_details.append({
            "title": v.get("title", ""),
            "sku": v.get("sku", ""),
            "available": v.get("available", False),
            "price": v.get("price", ""),
            "grams": v.get("grams", 0),
        })

    meta: dict[str, Any] = {
        "product_id": product.get("id"),
        "handle": product.get("handle"),
        "vendor": product.get("vendor"),
        "sku_list": all_skus,
        "options": option_names,
        "option_values": option_values,
        "variants_count": len(variants),
        "any_variant_available": all_available,
        "tags": product.get("tags", ""),
        "created_at": product.get("created_at"),
        "updated_at": product.get("updated_at"),
        "published_at": product.get("published_at"),
        "scrape_timestamp": scrape_timestamp,
    }

    # Add variant_details separately for readability
    meta["variant_details"] = variant_details

    import json
    return json.dumps(meta, ensure_ascii=False, default=str)


# ── Main parser ─────────────────────────────────────────────────────────────


def parse_product(product_json: dict) -> Optional[dict]:
    """
    Parse a Shopify product JSON dict into the Finds products table schema.

    Returns a dict ready for Supabase upsert, or None if critical fields
    are missing.
    """
    try:
        import datetime

        title = product_json.get("title", "")
        if not title:
            logger.warning("Product has no title, skipping.")
            return None

        handle = product_json.get("handle", "")
        if not handle:
            logger.warning("Product has no handle, skipping.")
            return None

        product_url = f"https://doubleyou-studios.com/products/{handle}"
        variants = product_json.get("variants", [])
        images = product_json.get("images", [])
        body_html = product_json.get("body_html", "")
        tags_raw = product_json.get("tags", "")

        # ── Clean description ───────────────────────────────────────────
        # Remove HTML tags from body_html
        description = re.sub(r"<[^>]+>", "", body_html).strip()
        description = re.sub(r"\s+", " ", description)
        if description == " ":
            description = ""

        # ── Images ──────────────────────────────────────────────────────
        front_url, back_url, gallery = _classify_product_images(images)
        if not front_url:
            logger.warning("No front image for %s, skipping.", title)
            return None

        # ── Pricing ─────────────────────────────────────────────────────
        price = _parse_price(variants)
        sale = _parse_sale_price(variants)

        # ── Category ────────────────────────────────────────────────────
        product_type = product_json.get("product_type", "") or ""
        category = _derive_category(title, product_type)

        # ── Tags ────────────────────────────────────────────────────────
        if isinstance(tags_raw, str):
            tags_list = (
                [t.strip() for t in tags_raw.split(",") if t.strip()]
                if tags_raw
                else []
            )
        elif isinstance(tags_raw, list):
            tags_list = tags_raw
        else:
            tags_list = []

        # ── Size ────────────────────────────────────────────────────────
        # Extract available sizes from variants
        sizes = [v.get("title", "") for v in variants]
        size_str = ", ".join(sizes) if sizes else None

        # ── Additional images ───────────────────────────────────────────
        additional_images = " , ".join(gallery) if gallery else None

        # ── Metadata ────────────────────────────────────────────────────
        scrape_ts = datetime.datetime.utcnow().isoformat() + "Z"
        metadata = build_metadata(
            product_json, variants, scrape_ts
        )

        # ── ID ──────────────────────────────────────────────────────────
        pid = generate_product_id(SOURCE, product_url)

        # ── Assemble row ────────────────────────────────────────────────
        row = {
            "id": pid,
            "source": SOURCE,
            "product_url": product_url,
            "affiliate_url": None,
            "image_url": front_url,
            "compressed_image_url": None,
            "back_image_url": back_url,
            "brand": BRAND,
            "title": title,
            "description": description or None,
            "category": category,
            "gender": "unisex",
            "price": price or None,
            "sale": sale,
            "metadata": metadata,
            "size": size_str,
            "second_hand": SECOND_HAND,
            "country": None,
            "tags": tags_list if tags_list else None,
            "additional_images": additional_images,
            "other": None,
            "image_embedding": None,
            "back_image_embedding": None,
            "info_embedding": None,
            "created_at": None,  # DB handles default
        }

        return row

    except Exception as exc:
        logger.error("Error parsing product %s: %s",
                      product_json.get("title", "?"), exc)
        return None
