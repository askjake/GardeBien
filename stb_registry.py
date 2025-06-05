"""
stb_registry.py
───────────────────────────────────────────────────────────────────────────────
Central registry for mapping a logical STB-ID (e.g. "A")
to both its:

• capture device string (ffmpeg DirectShow or /dev/video path)
• DP-Studio PortX number   (string like "8")

Nothing else in the pipeline should hard-code these strings; import this
module instead.
"""

from __future__ import annotations
from typing import Dict

# ════════════════════════════════════════════════════════════════════════════
# 1.  Static tables provided by the lab
# ════════════════════════════════════════════════════════════════════════════

# Index → DirectShow device friendly-name
_DEVICE_BY_INDEX: Dict[int, str] = {
    1:  "Video (00-0 Pro Capture Quad HDMI)",
    2:  "Video (00-1 Pro Capture Quad HDMI)",
    3:  "Video (00-2 Pro Capture Quad HDMI)",
    4:  "Video (00-3 Pro Capture Quad HDMI)",
    5:  "Video (01-0 Pro Capture Quad HDMI)",
    6:  "Video (01-1 Pro Capture Quad HDMI)",
    7:  "Video (01-2 Pro Capture Quad HDMI)",
    8:  "Video (01-3 Pro Capture Quad HDMI)",
    9:  "Video (02-0 Pro Capture Quad HDMI)",
    10: "Video (02-1 Pro Capture Quad HDMI)",
    11: "Video (02-2 Pro Capture Quad HDMI)",
    12: "Video (02-3 Pro Capture Quad HDMI)",
    13: "Video (03-0 Pro Capture Quad HDMI)",
    14: "Video (03-1 Pro Capture Quad HDMI)",
    15: "Video (03-2 Pro Capture Quad HDMI)",
    16: "Video (03-3 Pro Capture Quad HDMI)",
}

# Bit-mask → capture-index   (comes from CARD_MAP in dp_lib.py)
_CAPTURE_IDX_BY_MASK: Dict[int, int] = {
     1:  1,   2:  2,   3:  4,    4:8,
    5: 16,  6:  32,  7: 64,  8:128,
   9:256, 10:512, 11:1024, 12:2048,
  13:4096, 14:5192, 15:16384, 16:32768,   
}

# translate any "0" index to 16 for the device table
_CAPTURE_IDX_BY_MASK = {
    mask: (16 if idx == 0 else idx)
    for mask, idx in _CAPTURE_IDX_BY_MASK.items()
}

# ════════════════════════════════════════════════════════════════════════════
# 2.  Runtime mutable maps – populated by register_* helpers
# ════════════════════════════════════════════════════════════════════════════

CAPTURE_MAP  : Dict[str, str] = {}   # stb_id → ffmpeg device string
PORTVIEW_MAP : Dict[str, str] = {}   # stb_id → "8" etc.


# ════════════════════════════════════════════════════════════════════════════
# 3.  Helper API
# ════════════════════════════════════════════════════════════════════════════

def _dshow_str(name: str) -> str:
    """Return the string expected by ffmpeg on Windows."""
    return f'video={name}'

def capture_device_by_index(idx: int) -> str:
    if idx not in _DEVICE_BY_INDEX:
        raise KeyError(f"No capture device for index {idx}")
    return _dshow_str(_DEVICE_BY_INDEX[idx])

def capture_device_by_mask(mask: int) -> str:
    """Convert bit-mask (e.g. 128) → DirectShow string."""
    if mask not in _CAPTURE_IDX_BY_MASK:
        raise KeyError(f"Mask {mask} not in CARD_MAP")
    return capture_device_by_index(_CAPTURE_IDX_BY_MASK[mask])

# ---------------------------------------------------------------------------

def register_capture(stb_id: str, device: str):
    """Manually register a capture device string."""
    CAPTURE_MAP[stb_id] = device

def register_portview(stb_id: str, portview: str):
    PORTVIEW_MAP[stb_id] = str(portview)

def register_by_mask(stb_id: str, *, mask: int, portview: str):
    """
    Convenience: supply the bit-mask and portview; we look up the device
    name automatically.
    """
    dev = capture_device_by_mask(mask)
    register_capture(stb_id, dev)
    register_portview(stb_id, portview)

# ════════════════════════════════════════════════════════════════════════════
# 4.  Introspection helpers (diagnostics)
# ════════════════════════════════════════════════════════════════════════════

def summary() -> str:
    """Return a human-readable snapshot of current registry."""
    lines = ["STB_ID  Port  CaptureDevice"]
    for stb in sorted(set(CAPTURE_MAP) | set(PORTVIEW_MAP)):
        lines.append(f"{stb:<6}  {PORTVIEW_MAP.get(stb,'-'):<4}  {CAPTURE_MAP.get(stb,'-')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example usage/debug
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # demo – register logical "A" by bit-mask 128  (maps to device idx 4)
    register_by_mask("A", mask=128, portview="8")
    print(summary())
