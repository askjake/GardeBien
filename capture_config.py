# capture_config.py
import json, pathlib

CARD_MAP = {
    1: 2,    2: 9,   4: 8,   8: 15,
    16: 10, 32: 5,  64: 11, 128: 4,
    256: 3, 512: 13,1024:14,2048:12,
    4096: 6,8192: 1,16384: 7,32768: 0
}

DEVICES = json.loads(pathlib.Path("capture_devices.json").read_text())

def dshow_name_for_port(bit_mask: int) -> str:
    idx = CARD_MAP[bit_mask]          # 1‑based index from your dict
    dev_name = DEVICES[str(idx)]      # look up the saved name
    return f"video={dev_name}"
