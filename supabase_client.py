"""
Supabase client module for the Double You Studios scraper.

Handles:
- Batch upsert with (source, product_url) unique constraint
- Smart diffing against existing rows
- Stale product tracking via metadata.scrape_miss_count
- Stale product deletion after 2 consecutive misses
- Error handling with retry and logging
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from httpx import HTTPError

from config import BATCH_SIZE, SOURCE, SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Supabase REST client for products table operations."""

    def __init__(self) -> None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment."
            )
        self.url = SUPABASE_URL.rstrip("/")
        self.key = SUPABASE_KEY
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        self._client = httpx.Client(
            base_url=self.url,
            headers=self.headers,
            timeout=httpx.Timeout(30.0),
        )

    # ── Existing rows lookup ────────────────────────────────────────────

    def fetch_existing_rows(self) -> dict[str, dict]:
        """
        Fetch ALL existing rows for this source.
        Returns a dict keyed by product_url for fast lookup.
        """
        existing: dict[str, dict] = {}
        offset = 0
        limit = 1000

        while True:
            params = {
                "select": "*",
                "source": f"eq.{SOURCE}",
                "limit": limit,
                "offset": offset,
            }
            try:
                resp = self._client.get(
                    "/rest/v1/products",
                    params=params,
                )
                resp.raise_for_status()
                rows = resp.json()
                if not rows:
                    break
                for row in rows:
                    pu = row.get("product_url")
                    if pu:
                        existing[pu] = row
                offset += limit
                logger.debug(
                    "Fetched %d rows (offset %d).", len(rows), offset
                )
            except HTTPError as exc:
                logger.error("Failed to fetch existing rows: %s", exc)
                break

        logger.info("Loaded %d existing rows from DB.", len(existing))
        return existing

    # ── Smart diffing ───────────────────────────────────────────────────

    @staticmethod
    def _has_changed(new_row: dict, existing_row: dict) -> bool:
        """
        Deep compare scraped fields against the existing DB row.
        Returns True if ANY tracked field differs.
        """
        tracked_fields = [
            "title", "description", "price", "sale", "category",
            "gender", "image_url", "back_image_url", "additional_images",
            "metadata", "affiliate_url", "size", "tags", "country", "other",
        ]
        for field in tracked_fields:
            new_val = new_row.get(field)
            old_val = existing_row.get(field)
            # Normalize for comparison
            if field == "tags":
                # Compare as JSON strings
                new_str = json.dumps(new_val, sort_keys=True, default=str)
                old_str = json.dumps(old_val, sort_keys=True, default=str)
                if new_str != old_str:
                    return True
            elif field == "metadata":
                # Compare scraped metadata ignoring scrape_timestamp and
                # scrape_miss_count
                try:
                    new_md = json.loads(new_val) if isinstance(new_val, str) else (new_val or {})
                    old_md = json.loads(old_val) if isinstance(old_val, str) else (old_val or {})
                    # Remove volatile keys
                    for key in ("scrape_timestamp", "scrape_miss_count"):
                        new_md.pop(key, None)
                        old_md.pop(key, None)
                    if new_md != old_md:
                        return True
                except (json.JSONDecodeError, TypeError):
                    if new_val != old_val:
                        return True
            else:
                if new_val != old_val:
                    return True
        return False

    @staticmethod
    def _needs_image_embedding(new_row: dict, existing_row: dict) -> bool:
        """True if image_url changed (or product is new)."""
        return new_row.get("image_url") != existing_row.get("image_url")

    @staticmethod
    def _needs_back_embedding(new_row: dict, existing_row: dict) -> bool:
        """True if back_image_url changed (including added/removed)."""
        return new_row.get("back_image_url") != existing_row.get(
            "back_image_url"
        )

    @staticmethod
    def _needs_info_embedding(new_row: dict, existing_row: dict) -> bool:
        """True if any text field used for info text changed."""
        text_fields = [
            "title", "description", "category", "gender",
            "price", "sale", "metadata",
        ]
        for field in text_fields:
            if new_row.get(field) != existing_row.get(field):
                return True
        # Compare tags
        new_tags = new_row.get("tags")
        old_tags = existing_row.get("tags")
        if json.dumps(new_tags, sort_keys=True, default=str) != \
           json.dumps(old_tags, sort_keys=True, default=str):
            return True
        return False

    # ── Batch upsert ────────────────────────────────────────────────────

    def upsert_batch(
        self, rows: list[dict]
    ) -> tuple[int, int, list[str]]:
        """
        Upsert a batch of rows using the (source, product_url) unique key.

        Returns (upserted_count, failed_count, failed_ids).
        Uses the Supabase REST API with POST /rest/v1/products and
        on_conflict resolution.
        """
        if not rows:
            return 0, 0, []

        payload = []
        for row in rows:
            # Remove None values that shouldn't be sent
            clean = {k: v for k, v in row.items() if v is not None}
            # Always include source for the unique key
            clean["source"] = SOURCE
            payload.append(clean)

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                headers = {
                    **self.headers,
                    "Prefer": (
                        "return=minimal,"
                        "resolution=merge-duplicates"
                    ),
                }
                resp = self._client.post(
                    "/rest/v1/products",
                    params={
                        "on_conflict": "source,product_url",
                    },
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                return len(rows), 0, []
            except HTTPError as exc:
                logger.warning(
                    "Batch upsert attempt %d/%d failed: %s",
                    attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    backoff = 2 ** attempt
                    time.sleep(backoff)
                else:
                    failed_ids = [
                        r.get("id", "?") for r in rows
                    ]
                    logger.error(
                        "Batch upsert failed after %d attempts. "
                        "Failed IDs: %s",
                        max_retries, failed_ids,
                    )
                    return 0, len(rows), failed_ids

        return 0, len(rows), [r.get("id", "?") for r in rows]

    # ── Stale product management ────────────────────────────────────────

    def process_stale_products(
        self,
        seen_urls: set[str],
        existing_rows: dict[str, dict],
    ) -> tuple[list[str], list[str]]:
        """
        Process products not seen in this run.

        - Missed once → increment scrape_miss_count in metadata
        - Missed twice → DELETE from DB

        Returns (updated_ids, deleted_ids).
        """
        updated_ids: list[str] = []
        deleted_ids: list[str] = []

        for product_url, existing in existing_rows.items():
            if product_url in seen_urls:
                # Reset miss count if previously missed
                metadata_str = existing.get("metadata", "{}")
                try:
                    md = json.loads(metadata_str) if isinstance(
                        metadata_str, str
                    ) else (metadata_str or {})
                except (json.JSONDecodeError, TypeError):
                    md = {}

                if md.get("scrape_miss_count", 0) > 0:
                    md["scrape_miss_count"] = 0
                    md["scrape_reset_at"] = (
                        datetime.now(timezone.utc).isoformat()
                    )
                    existing_id = existing.get("id")
                    if existing_id:
                        self._update_metadata(existing_id, md)
                        updated_ids.append(existing_id)
                continue

            # Product NOT seen this run
            metadata_str = existing.get("metadata", "{}")
            try:
                md = json.loads(metadata_str) if isinstance(
                    metadata_str, str
                ) else (metadata_str or {})
            except (json.JSONDecodeError, TypeError):
                md = {}

            miss_count = md.get("scrape_miss_count", 0) + 1
            md["scrape_miss_count"] = miss_count
            md["scrape_missed_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

            existing_id = existing.get("id")
            if not existing_id:
                continue

            if miss_count >= 2:
                # Delete after 2 consecutive misses
                try:
                    self._client.delete(
                        f"/rest/v1/products",
                        params={"id": f"eq.{existing_id}"},
                    )
                    deleted_ids.append(existing_id)
                    logger.info(
                        "Deleted stale product %s (%s)",
                        existing.get("title", "?"), existing_id,
                    )
                except HTTPError as exc:
                    logger.error(
                        "Failed to delete %s: %s", existing_id, exc
                    )
            else:
                # Update miss count (first miss)
                self._update_metadata(existing_id, md)
                updated_ids.append(existing_id)
                logger.info(
                    "Marked product %s as missed (%d/2)",
                    existing.get("title", "?"), miss_count,
                )

        return updated_ids, deleted_ids

    def _update_metadata(self, product_id: str, metadata: dict) -> None:
        """Update the metadata JSON field for a product."""
        try:
            self._client.patch(
                f"/rest/v1/products",
                params={"id": f"eq.{product_id}"},
                json={"metadata": json.dumps(metadata, ensure_ascii=False)},
            )
        except HTTPError as exc:
            logger.error("Failed to update metadata for %s: %s",
                          product_id, exc)

    # ── Cleanup ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()
