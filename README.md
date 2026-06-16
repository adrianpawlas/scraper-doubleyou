# Double You Studios — Product Scraper

Scrapes all products from [doubleyou-studios.com](https://doubleyou-studios.com), generates SigLIP image embeddings (768-d) and text embeddings, and upserts into the **Finds** Supabase `products` table.

## Architecture

```
main.py              ← Entry point / orchestrator
config.py            ← Environment-based configuration
parser.py            ← Shopify JSON extraction & field mapping
embeddings.py        ← SigLIP image + text embedding pipeline
supabase_client.py   ← Batch upsert, smart diffing, stale cleanup
requirements.txt
.github/workflows/scrape.yml   ← GitHub Actions (weekly + manual)
```

### Data flow

1. **Discover** → paginate `/collections/all/products.json?page=N` until empty
2. **Fetch** → per product: `GET /products/{handle}.json`
3. **Parse** → map Shopify JSON → `products` table schema
4. **Diff** → compare scraped fields vs existing DB row; skip if unchanged
5. **Embed** → SigLIP for images, `all-mpnet-base-v2` for text (only when needed)
6. **Upsert** → batches of 50 via Supabase REST API
7. **Cleanup** → mark unseen products; delete after 2 consecutive misses

## Back-view detection

The parser scans all product images (excluding size-guide images) for filenames containing `Back` (case-insensitive). The first non-guide image matching this pattern is set as `back_image_url`.

Examples of detected back images:
- `SSTeeBlackEcom_Back.png` — Studio Sport Tee
- `TanPantBack_Ecom.png` — Cotton Twill Pant
- `RawDenimShortsEcom_Back.png` — Raw Denim Shorts

If no image contains "Back" in its filename, both `back_image_url` and `back_image_embedding` are set to NULL.

## Embedding pipeline

The user specification requested the HuggingFace free Inference API for SigLIP and Gemini embedding-001, but:
- `google/siglip-base-patch16-384` is **not available** on the free HF Inference API (returns 503)
- `Gemini embedding-001` is a **Google API** not hosted on HuggingFace

Therefore, the scraper runs both models **locally** using the `transformers` library:

| Embedding | Model | Dims | Method |
|-----------|-------|------|--------|
| `image_embedding` | `google/siglip-base-patch16-384` | 768 | Local `transformers` |
| `back_image_embedding` | Same SigLIP model | 768 | Local `transformers` |
| `info_embedding` | `sentence-transformers/all-mpnet-base-v2` | 768 | Local `sentence-transformers` |

All embeddings are L2-normalized before storage.

### Image pre-processing pipeline

1. Download image from CDN URL
2. Decode to RGB
3. Resize longest side to max 1280px (preserve aspect ratio, LANCZOS)
4. Re-encode as JPEG quality ~85 for consistency
5. Run through SigLIP model
6. L2-normalize output vector
7. Validate 768-d and finite values

## Rate limiting

- Store requests: 1.5s between requests (configurable via `SCRAPER_STORE_REQUEST_DELAY`)
- HF model calls: 0.5s between calls (configurable via `SCRAPER_HF_DELAY`)

## Stale product cleanup

Products not seen in a scrape run get their `metadata.scrape_miss_count` incremented:
- **Miss 0→1**: metadata updated with `scrape_miss_count: 1`
- **Miss 1→2**: product is **deleted** from the database
- **Seen again**: miss count is reset to 0

## Field formatting

| Field | Format |
|-------|--------|
| `price` / `sale` | `"1681.29CZK"` (number + currency code) |
| `additional_images` | `"url1 , url2 , url3"` (space-comma-space separated) |
| `category` | Derived from title keywords (e.g. `"Shirts"`, `"Knitwear"`, `"Bottoms"`) |
| `metadata` | JSON string with sizes, SKUs, availability, scrape timestamp |
| `id` | SHA-256 hash of `source:product_url` (stable across runs) |
| `tags` | PostgreSQL text array |
| `gender` | Always `unisex` |

## Local development

```bash
# Clone and enter
git clone https://github.com/adrianpawlas/scraper-doubleyou.git
cd scraper-doubleyou

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Supabase credentials

# Dry run (scrape + parse only, no DB writes)
python main.py --dry-run

# Full run
python main.py
```

## GitHub Actions

The scraper runs automatically:
- **Scheduled**: Every Sunday at 04:30 AM UTC
- **Manual**: Via `workflow_dispatch` from the Actions tab

Secrets required in GitHub:
- `SUPABASE_URL`
- `SUPABASE_KEY`

Failed batch upserts are logged to `logs/failed_products.log` and uploaded as a CI artifact.
