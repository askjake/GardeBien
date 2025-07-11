# dp_dispatcher.py
import logging
import threading
import time
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import dp_lib
from capture_and_detector import CaptureManager
import stb_registry as reg
from dataset_store import save_sample, BASE_DIR

# pick this up from env or default to your static server + datasets folder
DATASET_HTTP_ROOT = os.getenv(
    "DATASET_HTTP_ROOT",
    "http://172.20.110.197:8080/datasets"
).rstrip("/")

from ui_graph import UIGraph

logger = logging.getLogger("dp_dispatcher")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

_LOCK = threading.Lock()

def register_stb(stb_id: str, *, capture: str, portview: str):
    reg.register_capture(stb_id, capture)
    reg.register_portview(stb_id, portview)
    logger.info("Registered STB %s → %s  (portview=%s)", stb_id, capture, portview)

def _start_cap(stb_id: str, record: bool, pre: float) -> Optional[str]:
    if not record:
        return None
    device = reg.CAPTURE_MAP.get(stb_id)
    if not device:
        logger.warning("No capture device registered for %s", stb_id)
        return None
    return CaptureManager.start(device, pre_roll=pre)

def _stop_cap(sid: str) -> Tuple[Path, Path, Path]:
    return CaptureManager.stop(sid)


def send(
    cmd: str,
    stb_id: str,
    *,
    record: bool = True,
    pre_roll: float = .3,
    add_to_dataset: bool = True,
    **kw,
):
    sess_id = _start_cap(stb_id, record, pre_roll)
    t0 = time.time()

    result = dp_lib.send_cmd(cmd, stb_id, **kw)

    latency = (time.time() - t0) * 1_000
    logger.info("Sent %-5s  → %-2s  %.1f ms", cmd, stb_id, latency)

    if sess_id and add_to_dataset:
        try:
            clip, bf, af = _stop_cap(sess_id)
            if bf.exists() and af.exists():
                meta = {"stb": stb_id, "latency_ms": int(latency)}

                # 1) save to disk
                rec = save_sample(clip, bf, af, cmd, meta)
                logger.info("Saved sample bf=%s af=%s clip=%s", bf, af, clip)

                # 2) build HTTP URLs based on BASE_DIR + relative path
                bf_path = Path(rec["source_files"]["bf"])
                af_path = Path(rec["source_files"]["af"])

                rel_bf = bf_path.relative_to(BASE_DIR)   # e.g. "20250708_xxx_RIGHT/bf.jpg"
                rel_af = af_path.relative_to(BASE_DIR)

                bf_url = f"{DATASET_HTTP_ROOT}/{rel_bf.as_posix()}"
                af_url = f"{DATASET_HTTP_ROOT}/{rel_af.as_posix()}"

                # 3) record in Neo4j with the correct URLs
                try:
                    with UIGraph() as g:
                        g.add_transition(
                            rec["bf_hash"],
                            rec["af_hash"],
                            cmd,
                            {"latency_ms": int(latency)},
                            bf_img_url=bf_url,
                            af_img_url=af_url,
                        )
                    logger.info(
                        "Graph transition added: %s → %s via %s",
                        rec["bf_hash"], rec["af_hash"], cmd
                    )
                except Exception as e:
                    logger.exception("Graph update failed: %s", e)

            else:
                logger.warning(
                    "Skipped dataset – missing bf/af: bf.exists=%s af.exists=%s",
                    bf.exists(), af.exists()
                )
        except Exception as e:
            logger.exception("Dataset save failed: %s", e)

    return result
