"""dataset_store.py
 Utility module for persisting (BF, AF, CMD)

 High‑level contract
 -------------------
 save_pair(bf_path, af_path, cmd, meta) -> dict
     • Copies the two frames into a timestamped sub‑folder
     • Calculates SHA‑1 hashes (stable node‑IDs)
     • Writes a meta.json with hashes + extra metadata
     • Returns the JSON record so caller can push to Neo4j immediately.

 iter_batches(batch_size) -> generator[list[tuple[pathlib.Path, pathlib.Path, str]]]
     • Yields batches so training jobs can stream from disk without loading
       everything into memory.

 The layout on disk ends up like::

     datasets/
         20240522_083620_123_RIGHT/
             bf.jpg
             af.jpg
             meta.json
         20240522_083703_987_RIGHT/
             bf.jpg
             af.jpg
             meta.json

 The module is intentionally dependency‑light (std lib only) so it can be
 imported inside DP Studio without extra wheels.
 """

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Tuple
import os, socket, uuid

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# These could become environment variables or be patched by the test‑runner.
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------
# Global roots – override with env vars so each machine can point
# wherever it likes without touching code.
#   • GB_DATASETS   – where BF/AF/clip folders live
#   • GB_MODELS     – (optional) where .pt weights land after training
# ---------------------------------------------------------------------
BASE_DIR = Path(os.getenv("GB_DATASETS", r"Z:\datasets")).expanduser()
#BASE_DIR = Path("datasets")  # relative to project root
HOST_TAG = socket.gethostname().split(".")[0].upper()   # e.g. “CAPTURE-01”

BF_NAME = "bf.jpg"
AF_NAME = "af.jpg"
META_NAME = "meta.json"
DATASET_ROOT = Path("datasets")
CLIP_NAME = "clip.mp4"
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_base() -> None:
    """Create the datasets/ folder on first use."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)


def _sha1(file_path: Path) -> str:
    """Return hex SHA‑1 digest of a file (used as node ID)."""
    h = hashlib.sha1()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_pair(
    bf_path: Path | str,
    af_path: Path | str,
    cmd: str,
    meta: dict | None = None,
) -> dict:
    """Persist a single (BF, AF, CMD) example and return its metadata dict."""

    _ensure_base()

    # Timestamped dir name keeps examples ordered and unique.
    ts_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    pair_dir = BASE_DIR / f"{ts_stamp}_{cmd.upper()}"
    pair_dir.mkdir()

    # Copy frames so that the original capture can be discarded later.
    bf_dest = pair_dir / BF_NAME
    af_dest = pair_dir / AF_NAME
    shutil.copy2(bf_path, bf_dest)
    shutil.copy2(af_path, af_dest)

    record = {
        "cmd": cmd.upper(),
        "bf_hash": _sha1(bf_dest),
        "af_hash": _sha1(af_dest),
        "source_files": {
            "bf": str(bf_dest),
            "af": str(af_dest),
        },
        "meta": meta or {},
    }

    with (pair_dir / META_NAME).open("w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2)

    return record

def save_sample(clip_path, bf_path, af_path, cmd, meta):
    # e.g. create folder named by date+hash, copy clip.mp4, bf.jpg, af.jpg
    from datetime import datetime, timezone

    _ensure_base()
    # 1. Build a timestamped directory name
    ts_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    uniq     = uuid.uuid4().hex[:6]                     # 6-char slug
    pair_dir = BASE_DIR / f"{ts_stamp}_{HOST_TAG}_{uniq}_{cmd.upper()}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy the raw clip plus before/after frames
    clip_dest = pair_dir / "clip.mp4"
    bf_dest   = pair_dir / BF_NAME
    af_dest   = pair_dir / AF_NAME
    shutil.copy2(clip_path, clip_dest)
    shutil.copy2(bf_path,   bf_dest)
    shutil.copy2(af_path,   af_dest)

    # 3. Compute SHA-1 hashes of the persisted frames
    bf_hash = _sha1(bf_dest)
    af_hash = _sha1(af_dest)

    # 4. Build the metadata record and write it out
    record = {
        "cmd": cmd.upper(),
        "bf_hash": bf_hash,
        "af_hash": af_hash,
        "source_files": {
            "clip": str(clip_dest),
            "bf":   str(bf_dest),
            "af":   str(af_dest),
        },
        "meta": meta or {},
    }
    with (pair_dir / META_NAME).open("w", encoding="utf-8") as fp:
        json.dump(record, fp, indent=2)

    return record

def iter_batches(dataset_dir: str = "datasets", batch_size: int = 32):
    """Yield batches of (bf_path, af_path, cmd) tuples for training.

    The generator walks *lazily* over the dataset directory so it can run on
    machines with limited RAM.
    """
    assert batch_size > 0, "batch_size must be >= 1"
    _ensure_base()

    batch: List[Tuple[Path, Path, str]] = []
    for sample_dir in sorted(BASE_DIR.iterdir()):
        if not sample_dir.is_dir():
            continue
        clip = sample_dir / CLIP_NAME
        bf = sample_dir / BF_NAME
        af = sample_dir / AF_NAME
        meta_file = sample_dir / META_NAME
        if not (clip.exists() and bf.exists() and af.exists() and meta_file.exists()):
            continue  # skip incomplete samples
        from json import JSONDecodeError
        try:
            with meta_file.open("r", encoding="utf-8") as fp:
                cmd = json.load(fp).get("cmd", "UNKNOWN")
        except (JSONDecodeError, FileNotFoundError) as e:
            # warn once and skip this sample
            print(f"[WARN] bad meta: {meta_file} ({e})")
            continue
        batch.append((clip, bf, af, cmd))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Stand‑alone test (run `python dataset_store.py bf.jpg af.jpg RIGHT`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python dataset_store.py <bf_img> <af_img> <cmd>")
        sys.exit(1)

    bf_img, af_img, cmd_str = sys.argv[1:]
    rec = save_pair(Path(bf_img), Path(af_img), cmd_str)
    print("Saved:", json.dumps(rec, indent=2))
