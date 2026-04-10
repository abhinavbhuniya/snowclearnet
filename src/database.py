"""
SnowClearNet – Supabase Database Integration
=============================================
Handles connection to Supabase and all CRUD operations
for processed image metadata.

Supabase table schema (create via SQL editor):

    CREATE TABLE IF NOT EXISTS processed_images (
        id          BIGSERIAL PRIMARY KEY,
        uploaded    TEXT      NOT NULL,
        processed   TEXT      NOT NULL,
        psnr        REAL,
        ssim        REAL,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );

    -- Enable Row Level Security (optional, disable for dev)
    -- ALTER TABLE processed_images ENABLE ROW LEVEL SECURITY;
"""

import os
from datetime import datetime, timezone

from supabase import create_client, Client


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Client | None = None


def get_client() -> Client:
    """Return a cached Supabase client, creating one on first call."""
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment / .env file."
            )
        _client = create_client(url, key)
    return _client


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

TABLE = "processed_images"


def save_record(uploaded: str, processed: str, psnr_val: float, ssim_val: float) -> dict:
    """Insert a new processing record and return the created row."""
    client = get_client()
    data = {
        "uploaded": uploaded,
        "processed": processed,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = client.table(TABLE).insert(data).execute()
    return result.data[0] if result.data else data


def get_all_records(limit: int = 50) -> list[dict]:
    """Fetch the most recent processing records."""
    client = get_client()
    result = (
        client.table(TABLE)
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []
