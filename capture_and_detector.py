# capture_and_detector.py
"""
Change-detector + ffmpeg capture utilities for Garde-Bien
========================================================
1. ChangeDetector  – light-weight CGI-change detector (diff-ratio)
2. CaptureManager  – spin-up ffmpeg, record N-second clip,
                     grab BF / AF JPEGs, expose thread-safe
                     start/stop helpers for dp_dispatcher.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# 1.  ChangeDetector
# ────────────────────────────────────────────────────────────────────────────
class ChangeDetector:
    def __init__(self,
                 mask: Optional[np.ndarray] = None,
                 diff_threshold: float = 0.02):
        self.mask = mask
        self.diff_threshold = diff_threshold
        self._last_gray: Optional[np.ndarray] = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=self.mask)
        return gray

    def changed(self, frame: np.ndarray) -> bool:
        cur = self._preprocess(frame)
        if self._last_gray is None:
            self._last_gray = cur
            return False
        diff   = cv2.absdiff(cur, self._last_gray)
        ratio  = (diff > 25).sum() / diff.size  # simple diff-ratio
        self._last_gray = cur
        return ratio > self.diff_threshold


# ────────────────────────────────────────────────────────────────────────────
# 2.  CaptureSession  (one ffmpeg clip)
# ────────────────────────────────────────────────────────────────────────────

def _screenshot(device:str, out_path:Path):
    """
    One-off frame grab straight from the capture device.
    Works while the long-running recorder is busy.
    """
    cmd = [
        _FFMPEG, "-y",
        "-f", "dshow" if os.name == "nt" else "v4l2",
        "-i", device,
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path),
    ]
    subprocess.run(cmd,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL,
                   check=True)

_FFMPEG        = shutil.which("ffmpeg") or "ffmpeg"
_CAPTURE_ROOT  = Path("captures")
_CAPTURE_ROOT.mkdir(exist_ok=True)

class CaptureSession:
    """One ffmpeg recording (raw.mp4 + bf/af JPEG)."""

    def __init__(self,
                 device: str,
                 pre_roll: float = 0.3,
                 clip_secs: float = 5.0) -> None:
        ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.dir        = _CAPTURE_ROOT / ts
        self.dir.mkdir(parents=True, exist_ok=True)

        self.device     = device
        self.pre_roll   = pre_roll
        self.clip_secs  = clip_secs

        self.clip_path  = self.dir / "raw.mp4"
        self._bf_path   = self.dir / "bf.jpg"
        self._af_path   = self.dir / "af.jpg"
        self.proc: Optional[subprocess.Popen] = None

    # ───────── ffmpeg helpers ──────────────────────────────────────────
    def _build_cmd(self) -> list[str]:
        return [
            _FFMPEG, "-y",
            "-f", "dshow" if os.name == "nt" else "v4l2",
            "-i", self.device,
            "-t", str(self.clip_secs),                # stop after N seconds
            "-c:v", "libx264", "-preset", "ultrafast",
            str(self.clip_path),
        ]

    def _extract_frame(self, out_path: Path, timestamp: float):
        """Pull single frame at `timestamp` (sec) from clip."""
        cmd = [
            _FFMPEG, "-y",
            "-ss", str(timestamp),
            "-i", str(self.clip_path),
            "-frames:v", "1",
            "-q:v", "2",                     # visually loss-less JPEG
            str(out_path),
        ]
        subprocess.run(cmd,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)

    # ───────── public lifecycle ────────────────────────────────────────
    def start(self):
        """Launch ffmpeg & grab BEFORE frame after `pre_roll` sec."""
        self.proc = subprocess.Popen(
            self._build_cmd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # give HDMI a moment to settle
        time.sleep(self.pre_roll)

        # ← NEW: screenshot comes straight from device, not the clip
        _screenshot(self.device, self._bf_path)

    def stop(self) -> Tuple[Path, Path, Path]:
        """Wait for clip, grab AFTER frame, return (clip,bf,af)."""
        self.proc.wait(timeout=self.clip_secs + 2)  # just wait for ffmpeg
        # seek a hair before the end to be safe
        self._extract_frame(self._af_path,
                           timestamp=max(0, self.clip_secs - 0.1))
        return self.clip_path, self._bf_path, self._af_path

    # ───────── properties (read-only) ─────────────────────────────────
    @property
    def bf_path(self) -> Path: return self._bf_path
    @property
    def af_path(self) -> Path: return self._af_path


# ────────────────────────────────────────────────────────────────────────────
# 3.  CaptureManager  (thread-safe registry)
# ────────────────────────────────────────────────────────────────────────────
class CaptureManager:
    _lock      = threading.Lock()
    _sessions: Dict[str, CaptureSession] = {}

    @classmethod
    def start(cls, device: str, pre_roll: float = 0.3,
              clip_secs: float = 5.0) -> str:
        with cls._lock:
            sess = CaptureSession(device, pre_roll, clip_secs)
            sess.start()
            sess_id = sess.dir.name          # use folder name as handle
            cls._sessions[sess_id] = sess
            return sess_id

    @classmethod
    def stop(cls, sess_id: str) -> Tuple[Path, Path, Path]:
        with cls._lock:
            sess = cls._sessions.pop(sess_id, None)
        if sess is None:
            raise ValueError(f"Unknown capture session '{sess_id}'")
        return sess.stop()


# ────────────────────────────────────────────────────────────────────────────
# 4.  CLI sanity check
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Quick 3-second capture smoke-test")
    ap.add_argument("device", help='ffmpeg device string, e.g. video="HDMI"')
    args = ap.parse_args()

    try:
        sid = CaptureManager.start(args.device, pre_roll=0.3, clip_secs=3.0)
        print("Recording…")
        time.sleep(3.5)
        clip, bf, af = CaptureManager.stop(sid)
        print("✓ saved:", clip, bf, af)
    except Exception as e:
        sys.exit(f"✗ capture failed: {e}")
