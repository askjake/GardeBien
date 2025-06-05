import logging, threading, time
from pathlib import Path
from typing import Any, Optional, Tuple

import dp_lib
from capture_and_detector import CaptureManager
import stb_registry as reg               # NEW
from dataset_store import save_sample
logger = logging.getLogger("dp_dispatcher")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

_LOCK = threading.Lock()

def register_stb(stb_id: str, *, capture: str, portview: str):
    """One unified call from your bootstrap."""
    reg.register_capture(stb_id, capture)
    reg.register_portview(stb_id, portview)
    logger.info("Registered STB %s → %s  (portview=%s)", stb_id, capture, portview)

# ---------- internal helpers -------------------------------------------------
def _start_cap(stb_id: str, record: bool, pre: float) -> Optional[str]:
    if not record:
        return None
    device = reg.CAPTURE_MAP.get(stb_id)
    if not device:
        logger.warning("No capture device registered for %s", stb_id)
        return None
    return CaptureManager.start(device, pre_roll=pre)

def _stop_cap(sid: str) -> Tuple[Path, Path, Path]:
    """Stop the ffmpeg capture and return (clip, bf, af)."""
    clip, bf, af = CaptureManager.stop(sid)
    return clip, bf, af
# -----------------------------------------------------------------------------

def send(
    cmd: str,
    stb_id: str,
    *,
    record: bool = True,
    pre_roll: float = .3,
    add_to_dataset: bool = True,     # ← put it here
    **kw,
):
    sess_id = _start_cap(stb_id, record, pre_roll)
    t0 = time.time()

    result   = dp_lib.send_cmd(cmd, stb_id, **kw)

    latency = (time.time() - t0) * 1_000
    logger.info("Sent %-5s  → %-2s  %.1f ms", cmd, stb_id, latency)

    # 3) Stop capture & persist pair
    if sess_id and add_to_dataset:
        try:
            clip, bf, af = _stop_cap(sess_id)
            if bf.exists() and af.exists():
                meta = {"stb": stb_id, "latency_ms": int(latency)}
                # your save_sample(clip, bf, af, cmd, meta) should copy all three
                save_sample(clip, bf, af, cmd, meta)
                logger.info("Saved sample bf=%s af=%s clip=%s", bf, af, clip)
            else:
                logger.warning("Skipped dataset – missing bf/af: bf.exists=%s af.exists=%s", bf.exists(), af.exists())
        except Exception as e:
            logger.exception("Dataset save failed: %s", e)

    return result