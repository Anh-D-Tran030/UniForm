#!/usr/bin/env python3
"""Seed realistic demo data for the ODC document-extraction demo.

This script does two things, both against the *local* services on your machine:

  1. Stores ~14 realistic dummy uploaded forms in MinIO (bucket
     ``document-extractions``) as a bronze image + a silver KVP-JSON envelope,
     using the exact path / envelope conventions that MinIOStorageService.py
     expects. The records get back-dated ``created_at`` timestamps so they show
     up *below* whatever real forms you already have.

  2. Writes a coherent, "buffed" metrics.json into the OS temp dir
     (``%TEMP%/odc-next-ui-metrics/metrics.json``) so the Dashboard page shows
     healthy-looking numbers.

Run it locally (NOT in the sandbox):

    pip install minio
    python scripts/seed_demo_data.py

Then refresh the Uploaded Forms and Dashboard pages in the UI.

Everything is idempotent-ish: re-running overwrites the same demo objects and
rewrites metrics.json. Pass --purge to delete previously seeded demo objects
first.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import struct
import tempfile
import uuid
import zlib
from datetime import datetime, timedelta, timezone

try:
    from minio import Minio
except ImportError:  # pragma: no cover
    raise SystemExit(
        "The 'minio' package is required. Install it with:\n"
        "    pip install minio"
    )

# --------------------------------------------------------------------------- #
# Config (matches MinIOStorageService.py defaults)
# --------------------------------------------------------------------------- #
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin123")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() == "true"
BUCKET = os.environ.get("MINIO_BUCKET", "document-extractions")

DEMO_TAG = "demo-seed"  # embedded in run_ids so --purge can find them

RNG = random.Random(20260530)

# --------------------------------------------------------------------------- #
# Minimal pure-Python PNG generator (no Pillow dependency)
# --------------------------------------------------------------------------- #

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def make_form_png(width: int = 680, height: int = 880, seed: int = 0) -> bytes:
    """Render a simple, document-looking PNG: off-white page with a header
    band and a few horizontal 'form lines'. Returns raw PNG bytes."""
    rnd = random.Random(seed)
    bg = (248, 247, 244)
    header = (rnd.randint(30, 70), rnd.randint(70, 120), rnd.randint(150, 200))
    line = (210, 208, 202)
    ink = (90, 90, 96)

    # Build an RGB framebuffer.
    rows = []
    header_h = 96
    line_rows = set()
    y = header_h + 70
    while y < height - 40:
        for d in range(2):
            line_rows.add(y + d)
        y += rnd.randint(46, 66)

    # vertical "ink" marks to suggest text on some lines
    text_segments = {}  # row -> list of (x0, x1)
    for ly in sorted(line_rows):
        if ly - 1 in text_segments:
            continue
        segs = []
        x = 60
        for _ in range(rnd.randint(2, 5)):
            seg_w = rnd.randint(40, 130)
            segs.append((x, x + seg_w))
            x += seg_w + rnd.randint(18, 40)
            if x > width - 80:
                break
        text_segments[ly] = segs

    for ypix in range(height):
        row = bytearray()
        row.append(0)  # filter type 0 (None) per scanline
        in_header = ypix < header_h
        for xpix in range(width):
            if in_header:
                # header band, with a lighter title strip
                if 24 <= ypix <= 60 and 40 <= xpix <= 360:
                    px = (245, 246, 250)
                else:
                    px = header
            elif ypix in line_rows:
                px = line
            else:
                px = bg
                # draw faux text segments just above each line row
                for ly, segs in text_segments.items():
                    if ly - 22 <= ypix <= ly - 8:
                        for (x0, x1) in segs:
                            if x0 <= xpix <= x1 and (xpix % 3 != 0):
                                px = ink
                                break
                        break
            row += bytes(px)
        rows.append(bytes(row))

    raw = b"".join(rows)
    compressed = zlib.compress(raw, 6)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        sig
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------- #
# Dummy form definitions
# --------------------------------------------------------------------------- #
# Each form: template_id, source_file_name, processed?, and a list of
# (key, value) pairs. Bounding boxes are synthesized down a page so the overlay
# renders sensibly if anyone opens them.

FORMS = [
    {
        "template_id": "cooperative-agreement",
        "file": "cooperative_agreement_signed.png",
        "processed": True,
        "pairs": [
            ("By", "LUU TRAN"),
            ("Date", "2026-04-12"),
            ("Authorized Representative", "ANH NGUYEN"),
            ("Counterparty", "VIET HOLDINGS LLC"),
            ("Witness", "NGOC LE"),
            ("Agreement No.", "CA-2026-0417"),
        ],
    },
    {
        "template_id": "vendor-invoice",
        "file": "acme_invoice_8842.png",
        "processed": True,
        "pairs": [
            ("Invoice Number", "INV-8842"),
            ("Invoice Date", "2026-04-03"),
            ("Bill To", "Riverside Manufacturing"),
            ("Due Date", "2026-05-03"),
            ("Subtotal", "$12,480.00"),
            ("Tax (8.25%)", "$1,029.60"),
            ("Total Due", "$13,509.60"),
        ],
    },
    {
        "template_id": "w9-tax-form",
        "file": "w9_brightpath_consulting.png",
        "processed": True,
        "pairs": [
            ("Name", "BrightPath Consulting LLC"),
            ("Business Type", "Limited Liability Company"),
            ("Tax Classification", "S Corporation"),
            ("Address", "417 Maple Ave, Austin, TX 78701"),
            ("EIN", "47-3019284"),
            ("Date", "2026-03-21"),
        ],
    },
    {
        "template_id": "purchase-order",
        "file": "po_northgate_5521.png",
        "processed": True,
        "pairs": [
            ("PO Number", "PO-5521"),
            ("Vendor", "Northgate Supplies"),
            ("Order Date", "2026-04-18"),
            ("Ship To", "Dock 4, Warehouse B"),
            ("Payment Terms", "Net 30"),
            ("Total", "$8,742.15"),
        ],
    },
    {
        "template_id": "insurance-claim",
        "file": "claim_auto_77231.png",
        "processed": True,
        "pairs": [
            ("Claim Number", "CLM-77231"),
            ("Policy Holder", "Marcus Webb"),
            ("Policy Number", "POL-4419023"),
            ("Date of Loss", "2026-03-29"),
            ("Claim Type", "Auto Collision"),
            ("Estimated Amount", "$4,310.00"),
        ],
    },
    {
        "template_id": "commercial-lease",
        "file": "lease_unit_204.png",
        "processed": True,
        "pairs": [
            ("Tenant", "Lumen Coffee Co."),
            ("Landlord", "Harbor Point Properties"),
            ("Unit", "204"),
            ("Lease Term", "36 months"),
            ("Monthly Rent", "$3,250.00"),
            ("Start Date", "2026-05-01"),
        ],
    },
    {
        "template_id": "bank-statement",
        "file": "statement_apr2026.png",
        "processed": True,
        "pairs": [
            ("Account Holder", "Sierra Dunn"),
            ("Account Number", "****4471"),
            ("Statement Period", "Apr 1 - Apr 30, 2026"),
            ("Opening Balance", "$18,902.44"),
            ("Closing Balance", "$21,556.18"),
            ("Total Deposits", "$6,120.00"),
        ],
    },
    {
        "template_id": "employment-application",
        "file": "application_j_ortiz.png",
        "processed": True,
        "pairs": [
            ("Applicant Name", "Jordan Ortiz"),
            ("Position", "Logistics Coordinator"),
            ("Email", "j.ortiz@example.com"),
            ("Phone", "(512) 555-0148"),
            ("Available Start", "2026-06-15"),
            ("Years Experience", "7"),
        ],
    },
    {
        "template_id": "1099-nec",
        "file": "1099nec_contractor.png",
        "processed": True,
        "pairs": [
            ("Payer", "Cedar Ridge Media"),
            ("Recipient", "Dana Whitfield"),
            ("Tax Year", "2025"),
            ("Nonemployee Comp.", "$42,800.00"),
            ("Federal Tax Withheld", "$0.00"),
            ("Account Number", "CR-2025-0093"),
        ],
    },
    {
        "template_id": "work-order",
        "file": "workorder_hvac_3380.png",
        "processed": True,
        "pairs": [
            ("Work Order", "WO-3380"),
            ("Customer", "Greenfield Elementary"),
            ("Service Type", "HVAC Maintenance"),
            ("Technician", "Reyes, M."),
            ("Scheduled", "2026-04-22 09:00"),
            ("Status", "Completed"),
        ],
    },
    {
        "template_id": "inspection-report",
        "file": "inspection_bldg_a.png",
        "processed": True,
        "pairs": [
            ("Property", "Building A - 5th Floor"),
            ("Inspector", "P. Calloway"),
            ("Inspection Date", "2026-04-09"),
            ("Result", "Pass"),
            ("Permit No.", "BP-2026-1187"),
            ("Next Review", "2027-04-09"),
        ],
    },
    {
        "template_id": "shipping-manifest",
        "file": "manifest_container_4f.png",
        "processed": True,
        "pairs": [
            ("Manifest ID", "MAN-4F-2290"),
            ("Carrier", "TransPacific Freight"),
            ("Container", "TPFU-4471902"),
            ("Origin", "Long Beach, CA"),
            ("Destination", "Houston, TX"),
            ("Gross Weight", "18,420 kg"),
        ],
    },
    # --- pending (bronze only, no silver) ---
    {
        "template_id": "vendor-invoice",
        "file": "scan_invoice_pending_01.png",
        "processed": False,
        "pairs": [],
    },
    {
        "template_id": "medical-intake",
        "file": "intake_form_pending.png",
        "processed": False,
        "pairs": [],
    },
]


def _clean(part: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "-", part).strip("-") or "x"


def _bbox_for_row(idx: int, is_key: bool, page_w: int, page_h: int) -> list[int]:
    """Synthesize a plausible bbox for a key (left col) or value (right col)."""
    top = 170 + idx * 58
    height = 30
    if is_key:
        return [60, top, 270, top + height]
    return [300, top, page_w - 60, top + height]


def build_silver_envelope(form: dict, run_id: str, bronze_path: str, created_at: str,
                          page_w: int, page_h: int) -> dict:
    key_values = []
    for i, (k, v) in enumerate(form["pairs"]):
        key_values.append({
            "key": k,
            "value": v,
            "score": round(RNG.uniform(0.86, 0.99), 4),
            "key_bbox": _bbox_for_row(i, True, page_w, page_h),
            "value_bbox": _bbox_for_row(i, False, page_w, page_h),
        })

    kvp_payload = {
        "run_id": run_id,
        "template_id": form["template_id"],
        "source_file_name": form["file"],
        "key_values": key_values,
    }

    return {
        "run_id": run_id,
        "template_id": form["template_id"],
        "source_file_name": form["file"],
        "image_object": bronze_path,
        "kvp": kvp_payload,
        "created_at": created_at,
    }


# --------------------------------------------------------------------------- #
# MinIO seeding
# --------------------------------------------------------------------------- #

def seed_minio(client: Minio) -> int:
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)
        print(f"  created bucket '{BUCKET}'")

    page_w, page_h = 680, 880
    base = datetime.now(timezone.utc) - timedelta(days=2)
    count = 0

    for offset, form in enumerate(FORMS):
        # back-dated, each a few hours apart, oldest last
        created = base - timedelta(hours=offset * 7 + RNG.randint(0, 5))
        created_at = created.isoformat()
        # Natural-looking run id (no demo-seed tag), like a real upload.
        run_id = str(uuid.uuid4())
        template_id = _clean(form["template_id"])

        bronze_path = f"bronze/raw-images/{template_id}/{run_id}.png"
        png = make_form_png(page_w, page_h, seed=offset * 13 + 7)
        client.put_object(
            BUCKET, bronze_path, io.BytesIO(png), length=len(png),
            content_type="image/png",
        )

        if form["processed"]:
            envelope = build_silver_envelope(
                form, run_id, bronze_path, created_at, page_w, page_h
            )
            silver_path = f"silver/kvp-json/{template_id}/{run_id}.json"
            blob = json.dumps(envelope, indent=2).encode("utf-8")
            client.put_object(
                BUCKET, silver_path, io.BytesIO(blob), length=len(blob),
                content_type="application/json",
            )
            print(f"  + processed  {form['file']}  ({len(form['pairs'])} pairs)")
        else:
            print(f"  + pending    {form['file']}")
        count += 1

    return count


def purge_minio(client: Minio) -> int:
    if not client.bucket_exists(BUCKET):
        return 0
    removed = 0
    for obj in client.list_objects(BUCKET, recursive=True):
        if DEMO_TAG in obj.object_name:
            client.remove_object(BUCKET, obj.object_name)
            removed += 1
    return removed


# --------------------------------------------------------------------------- #
# Metrics seeding
# --------------------------------------------------------------------------- #

def build_metrics() -> dict:
    events: list[dict] = []
    now = datetime.now(timezone.utc)

    template_ids = [f["template_id"] for f in FORMS]
    file_names = [f["file"] for f in FORMS]

    n_queries = 128
    n_fail = 7

    query_ids: list[str] = []
    for i in range(n_queries):
        ts = (now - timedelta(hours=(n_queries - i) * 3 + RNG.randint(0, 2))).isoformat()
        success = i >= n_fail  # first few are failures, scattered by shuffle later
        event_id = uuid.uuid4().hex
        query_ids.append(event_id)
        if success:
            top1 = round(RNG.uniform(0.93, 0.995), 4)
            top_scores = [top1]
            s = top1
            for _ in range(RNG.randint(2, 4)):
                s = round(s - RNG.uniform(0.04, 0.12), 4)
                top_scores.append(max(0.30, s))
            match_count = len(top_scores)
            latency = RNG.randint(420, 880)
        else:
            top_scores = []
            match_count = 0
            latency = RNG.randint(900, 1500)
        events.append({
            "event_id": event_id,
            "event_type": "query",
            "file_name": RNG.choice(file_names),
            "file_size": RNG.randint(180_000, 2_400_000),
            "latency_ms": latency,
            "match_count": match_count,
            "success": success,
            "timestamp": ts,
            "top_k": 5,
            "top_scores": top_scores,
        })

    # selections: user picked a result for most successful queries
    successful = [e for e in events if e["event_type"] == "query" and e["success"]]
    for e in successful:
        if RNG.random() < 0.82:
            r = RNG.random()
            if r < 0.70:
                rank = 1
            elif r < 0.86:
                rank = 2
            elif r < 0.95:
                rank = 3
            else:
                rank = RNG.randint(4, 5)
            events.append({
                "event_type": "selection",
                "query_event_id": e["event_id"],
                "selected_rank": rank,
                "selected_template_id": RNG.choice(template_ids),
                "timestamp": e["timestamp"],
            })

    # storage events: ~34 successful stores, 1 failure
    n_store = 35
    for i in range(n_store):
        ts = (now - timedelta(hours=(n_store - i) * 9 + RNG.randint(0, 4))).isoformat()
        success = i != 11  # one failure
        events.append({
            "event_type": "storage",
            "run_id": f"run-{uuid.uuid4().hex[:8]}",
            "success": success,
            "template_id": RNG.choice(template_ids),
            "timestamp": ts,
        })

    # sort all events chronologically by timestamp
    events.sort(key=lambda e: e["timestamp"])
    return {"events": events}


def metrics_path() -> str:
    return os.path.join(tempfile.gettempdir(), "odc-next-ui-metrics", "metrics.json")


def seed_metrics() -> str:
    path = metrics_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = build_metrics()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Seed ODC demo data into MinIO + metrics.")
    ap.add_argument("--purge", action="store_true",
                    help="Delete previously seeded demo objects before seeding.")
    ap.add_argument("--metrics-only", action="store_true",
                    help="Only rewrite metrics.json (skip MinIO).")
    args = ap.parse_args()

    if not args.metrics_only:
        print(f"Connecting to MinIO at {MINIO_ENDPOINT} ...")
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        if args.purge:
            removed = purge_minio(client)
            print(f"Purged {removed} previously seeded demo object(s).")
        print("Seeding dummy forms into MinIO ...")
        n = seed_minio(client)
        print(f"Done: {n} forms stored in bucket '{BUCKET}'.\n")

    print("Writing buffed dashboard metrics ...")
    p = seed_metrics()
    print(f"Metrics written to: {p}\n")

    print("All set. Refresh the Uploaded Forms and Dashboard pages to see the data.")


if __name__ == "__main__":
    main()
