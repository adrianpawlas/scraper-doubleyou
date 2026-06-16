#!/usr/bin/env python3
"""
Double You Studios — Product Scraper

Scrapes every product from https://doubleyou-studios.com, extracts full
metadata, generates SigLIP image embeddings and text embeddings, and
upserts into the Finds Supabase products table.

Usage:
    python main.py                    # Full scrape
    python main.py --dry-run          # Scrape + print summary, no DB writes

Environment variables (or .env file):
    SUPABASE_URL
    SUPABASE_KEY
    HF_TOKEN                  (optional)
    SCRAPER_REQUEST_DELAY     (optional, default 1.0)
    SCRAPER_HF_DELAY          (optional, default 0.5)
    SCRAPER_BATCH_SIZE        (optional, default 50)
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BATCH_SIZE,
    EMBEDDING_VERSION,
    HF_DELAY,
    REQUEST_DELAY,
    SOURCE,
    STORE_REQUEST_DELAY,
)
from embeddings import (
    build_info_text,
    embed_image,
    embed_text,
)
from parser import (
    discover_all_product_urls,
    fetch_product_json,
    parse_product,
)
from supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


# ── Logging setup ───────────────────────────────────────────────────────────


def setup_logging(dry_run: bool = False) -> None:
    """Configure structured JSON logging for CI readability."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    log_level = logging.INFO
    if dry_run:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


# ── Run summary ─────────────────────────────────────────────────────────────


class RunSummary:
    """Tracks statistics for the final run summary."""

    def __init__(self) -> None:
        self.total_products = 0
        self.new_products = 0
        self.updated_products = 0
        self.unchanged_products = 0
        self.front_embeddings = 0
        self.back_embeddings = 0
        self.text_embeddings = 0
        self.stale_updated = 0
        self.stale_deleted = 0
        self.errors: list[str] = []
        self.failed_ids: list[str] = []

    def print(self) -> None:
        """Print the final run summary."""
        summary = f"""
╔══════════════════════════════════════════╗
║     Double You Studios — Scrape Summary  ║
╠══════════════════════════════════════════╣
║  Products found:          {self.total_products:>5}          ║
║  New products:            {self.new_products:>5}          ║
║  Products updated:        {self.updated_products:>5}          ║
║  Products unchanged:      {self.unchanged_products:>5}          ║
║  Front embeddings:        {self.front_embeddings:>5}          ║
║  Back embeddings:         {self.back_embeddings:>5}          ║
║  Text embeddings:         {self.text_embeddings:>5}          ║
║  Stale products marked:   {self.stale_updated:>5}          ║
║  Stale products deleted:  {self.stale_deleted:>5}          ║
║  Errors:                  {len(self.errors):>5}          ║
║  Failed upsert IDs:       {len(self.failed_ids):>5}          ║
╚══════════════════════════════════════════╝
"""
        print(summary)

        if self.errors:
            print("── Error log ──")
            for err in self.errors[-10:]:  # Show last 10
                print(f"  • {err}")

        if self.failed_ids:
            print("── Failed upsert IDs ──")
            for fid in self.failed_ids:
                print(f"  • {fid}")
            # Write to log file for CI artifact
            self._write_failed_log(self.failed_ids)

    @staticmethod
    def _write_failed_log(failed_ids: list[str]) -> None:
        """Write failed product IDs to a log file for CI artifacts."""
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "failed_products.log")
        with open(log_path, "w") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")
        logger.info(
            "Wrote %d failed product IDs to %s",
            len(failed_ids), log_path,
        )


# ── Download image helper ───────────────────────────────────────────────────


def download_image(
    url: str, client: httpx.Client
) -> Optional[bytes]:
    """Download image bytes from URL. Returns None on failure."""
    try:
        resp = client.get(url, timeout=30.0)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.error("Failed to download image %s: %s", url, exc)
        return None


# ── Main scrape function ────────────────────────────────────────────────────


def run_scrape(dry_run: bool = False, skip_embeddings: bool = False) -> RunSummary:
    """
    Execute the full scrape pipeline.

    Steps:
    1. Discover all product URLs via paginated collection API
    2. Fetch existing products from Supabase
    3. For each product: fetch JSON, parse, diff, embed, queue
    4. Batch upsert to Supabase
    5. Process stale products
    6. Print summary
    """
    summary = RunSummary()
    http_client = httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=15.0),
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; DoubleYouScraper/1.0; "
                "+https://github.com/adrianpawlas/scraper-doubleyou)"
            ),
        },
    )

    # ── Step 1: Discover product URLs ──────────────────────────────────
    logger.info("Step 1: Discovering product URLs...")
    product_urls = discover_all_product_urls()
    summary.total_products = len(product_urls)
    logger.info("Found %d products.", len(product_urls))

    if not product_urls:
        logger.warning("No products found. Exiting.")
        http_client.close()
        return summary

    # ── Step 2: Connect to Supabase ────────────────────────────────────
    supabase: Optional[SupabaseClient] = None
    existing_rows: dict[str, dict] = {}
    if not dry_run:
        try:
            supabase = SupabaseClient()
            existing_rows = supabase.fetch_existing_rows()
            logger.info(
                "Loaded %d existing rows from Supabase.", len(existing_rows)
            )
        except ValueError as exc:
            logger.error("Supabase config error: %s", exc)
            logger.info("Running in dry-run mode due to config error.")
            dry_run = True

    # ── Step 3: Process each product ───────────────────────────────────
    seen_urls: set[str] = set()
    upsert_queue: list[dict] = []

    for idx, product_url in enumerate(product_urls):
        logger.info(
            "[%d/%d] Processing: %s",
            idx + 1, len(product_urls), product_url,
        )

        # Fetch product JSON
        product_json = fetch_product_json(product_url)
        if not product_json:
            summary.errors.append(
                f"Failed to fetch product JSON: {product_url}"
            )
            continue

        # Parse into row format
        row = parse_product(product_json)
        if not row:
            summary.errors.append(
                f"Failed to parse product: {product_url}"
            )
            continue

        seen_urls.add(product_url)
        existing = existing_rows.get(product_url)
        is_new = existing is None

        # ── Determine what needs updating ──────────────────────────
        if is_new:
            summary.new_products += 1
            needs_image_embed = True
            needs_back_embed = (
                row.get("back_image_url") is not None
            )
            needs_info_embed = True
        else:
            changed = supabase._has_changed(
                row, existing
            ) if supabase else True
            if not changed:
                summary.unchanged_products += 1
                logger.debug("  No changes, skipping.")
                continue

            summary.updated_products += 1
            needs_image_embed = supabase._needs_image_embedding(
                row, existing
            ) if supabase else True
            needs_back_embed = supabase._needs_back_embedding(
                row, existing
            ) if supabase else (
                row.get("back_image_url") is not None
            )
            needs_info_embed = supabase._needs_info_embedding(
                row, existing
            ) if supabase else True

        # ── Generate embeddings (skip in dry-run mode) ──────────────
        if not skip_embeddings:
            front_url = row.get("image_url")
            back_url = row.get("back_image_url")

            if needs_image_embed and front_url:
                logger.debug("  Generating front image embedding...")
                img_data = download_image(front_url, http_client)
                if img_data:
                    embedding = embed_image(img_data)
                    if embedding is not None:
                        row["image_embedding"] = embedding.tolist()
                        row["embedding_version"] = EMBEDDING_VERSION
                        summary.front_embeddings += 1
                time.sleep(REQUEST_DELAY)

            if needs_back_embed and back_url:
                logger.debug("  Generating back image embedding...")
                img_data = download_image(back_url, http_client)
                if img_data:
                    embedding = embed_image(img_data)
                    if embedding is not None:
                        row["back_image_embedding"] = embedding.tolist()
                        summary.back_embeddings += 1
                time.sleep(REQUEST_DELAY)

            if needs_info_embed:
                logger.debug("  Generating text embedding...")
                info_text = build_info_text(row)
                if info_text:
                    embedding = embed_text(info_text)
                    if embedding is not None:
                        row["info_embedding"] = embedding.tolist()
                        summary.text_embeddings += 1
                time.sleep(HF_DELAY)

        # Queue for upsert
        upsert_queue.append(row)

        # ── Batch upsert ───────────────────────────────────────────
        if len(upsert_queue) >= BATCH_SIZE:
            _flush_batch(
                supabase, upsert_queue, summary, dry_run
            )
            upsert_queue = []

    # ── Final batch flush ───────────────────────────────────────────────
    if upsert_queue:
        _flush_batch(supabase, upsert_queue, summary, dry_run)

    # ── Step 4: Process stale products ─────────────────────────────────
    if supabase and not dry_run and existing_rows:
        updated_ids, deleted_ids = supabase.process_stale_products(
            seen_urls, existing_rows,
        )
        summary.stale_updated = len(updated_ids)
        summary.stale_deleted = len(deleted_ids)

    # ── Cleanup ────────────────────────────────────────────────────────
    http_client.close()
    if supabase:
        supabase.close()

    # ── Summary ─────────────────────────────────────────────────────────
    summary.print()
    return summary


def _flush_batch(
    supabase: Optional[SupabaseClient],
    batch: list[dict],
    summary: RunSummary,
    dry_run: bool,
) -> None:
    """Upsert a batch of rows, tracking results."""
    if dry_run or not supabase:
        logger.info(
            "[DRY RUN] Would upsert %d products.", len(batch)
        )
        return

    upserted, failed, failed_ids = supabase.upsert_batch(batch)
    if failed_ids:
        summary.failed_ids.extend(failed_ids)
    logger.info(
        "Batch upsert: %d upserted, %d failed.",
        upserted, failed,
    )


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Double You Studios product scraper"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and parse products but don't write to DB",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(dry_run=args.dry_run)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Double You Studios scraper (dry_run=%s)", args.dry_run)
    start = time.time()

    try:
        summary = run_scrape(dry_run=args.dry_run, skip_embeddings=args.dry_run)
    except Exception as exc:
        logger.exception("Unhandled exception during scrape: %s", exc)
        return 1

    elapsed = time.time() - start
    logger.info(
        "Scrape complete in %.1f seconds. %d errors.",
        elapsed, len(summary.errors),
    )

    # Return non-zero if there were failures
    if summary.errors or summary.failed_ids:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
