#!/usr/bin/env python3
"""
dp_lib.py – helpers for DP-Studio tests that pull frames directly from
capture cards (no VideoRecordAgent).  Python 3.10+, OpenCV with FFmpeg/DSHOW.
"""
from __future__ import annotations

import stb_registry as reg
import os, json, time, uuid, queue, threading, subprocess, math, tempfile, shutil
from datetime import datetime
from typing import Tuple, List, Optional
import requests, cv2, pytesseract, mysql.connector, xml.etree.ElementTree as ET

DESIRED_WIDTH  = 1920
DESIRED_HEIGHT = 1080

# ───────────── basic paths / config ─────────────
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

with open(r"C:/DPUnified/DPStudio-BackEnd/dpstudio-be/config.json") as fh:
    SERVER_IP = json.load(fh)["serverIP"]

MYSQL = dict(user="stbautomation", password="godp#123",
             host=SERVER_IP, database="devicepartner")

XML_PATH   = r"C:/DPUnified/Configs/UnifiedSettings.xml"
TOOLS_DIR  = r"C:/DPUnified/Tools"
VID_DIR    = rf"{TOOLS_DIR}/VideoRecordAgent"          # still used by latency helper
EVENT_WRITER_EXE = rf"{TOOLS_DIR}/EventLogWriter/EventLogWriter.exe"
CLASSIFY_EXE     = rf"{TOOLS_DIR}/ClassifyFailureAgent/ClassifyFailure.exe"

# ───────── capture-card map  ─────────
CARD_MAP: dict[int, object] = {
    1: 2,       # 1
    2: 9,       # 2
    4: 8,       # 3
    8: 15,       # 4
    16: 10,      # 5
    32: 5,      # 6
    64: 11,      # 7
    128: 4,     # 8
    256: 3,     # 9
    512: 13,     # 10
    1024: 14,   # 11
    2048: 12,   # 12
    4096: 6,   # 13
    8192: 1,   # 14
    16384: 7,  # 15
    32768: 0,  # 16
}




# ════════════════════════════════════════════════════════════════════════════
#                          DP-Unified helpers (unchanged)
# ════════════════════════════════════════════════════════════════════════════
def _xml_value(portview: str, key: str) -> Optional[str]:
    root = ET.parse(XML_PATH).getroot()
    for port in root:
        if port.attrib.get("name") == f"Port{portview}":
            for child in port:
                if child.attrib.get("name") == key:
                    return child.text
    return None


def send_command(portview: str, cmd: str):
    ct = _xml_value(portview, "CommandType")
    #print(f"[{datetime.now():%H:%M:%S.%f}] ▶ send_command "
    #      f"port={portview}  cmd={cmd}  type={ct}")
    timeout = float(os.getenv("DP_TIMEOUT", "5.0"))

    try:
        if cmd == "RESET_USER_SETTINGS":          # ← NEW special case
            exe  = "python.exe"
            args = [
                rf"{TOOLS_DIR}/DishSendRESTCommand/sgs_remote.py",
                "-i", _xml_value(portview, "RESTAPIServerIP"),
                "-s", _xml_value(portview, "RESTAPISTB"),
                "reset_stb_user_settings", "t0"
            ]
            subprocess.run([exe, *args],
                           cwd=rf"{TOOLS_DIR}/DishSendRESTCommand",
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout + 10)
            return                                 # done – no further cases

        if ct == "RF-IPCommand":
            requests.get(f"{_xml_value(portview,'RFURL')}/{cmd}/0",
                         timeout=timeout)
        elif ct == "RFCommand":
            requests.get(f"{_xml_value(portview,'RFIPAddress')}/{cmd}",
                         timeout=timeout)
        else:  # RESTAPICommand
            exe = "python.exe"
            args = [
                rf"{TOOLS_DIR}/DishSendRESTCommand/sgs_remote.py",
                "-i", _xml_value(portview,"RESTAPIServerIP"),
                "-s", _xml_value(portview,"RESTAPISTB"),
                cmd, "t0"
            ]
            subprocess.run([exe, *args],
                           cwd=rf"{TOOLS_DIR}/DishSendRESTCommand",
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout+2)
    except Exception as e:
        print(f"[WARN] send_command('{cmd}') failed on port {portview}: {e}")

def mask_to_single_port(mask:int) -> str:
    if mask == 0 or mask & (mask-1):
        raise ValueError("mask must have exactly one bit set")
    return str(mask.bit_length())

# ═════════════════════   live-capture frame grabber   ═══════════════════════
def _device_string(mask:int) -> str:
    if mask not in CARD_MAP:
        raise RuntimeError(f"No capture-card mapping for mask {mask}")
    return CARD_MAP[mask]

def grab_current_frame(mask: int,
                       retries: int = 5,
                       wait: float = 0.5) -> "np.ndarray":
    dev = _device_string(mask)

    # choose backend automatically
    if isinstance(dev, int):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
    else:                      # string e.g. "video=Decklink …"
        backends = [cv2.CAP_FFMPEG, cv2.CAP_DSHOW, cv2.CAP_MSMF]

    for att in range(1, retries + 1):
        for be in backends:
            cap = cv2.VideoCapture(dev, be)
            # force 16:9 resolution (adjust if your card is 720p)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DESIRED_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None and frame.size:
                return frame
        print(f"[WARN] no frame from {dev} (try {att}/{retries})")
        time.sleep(wait)

    raise RuntimeError(f"grab_current_frame: no good frame from {dev}")


# ═════════════════════   legacy MP4 reader (optional)   ═════════════════════
def _grab_frame_from_mp4(port:str, seek_ms:int = 5000) -> "np.ndarray":
    """
    Fallback helper if you really need to read the old VideoRecordAgent file.
    """
    src = rf"{VID_DIR}/record{port}/portview{port}.mp4"
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    shutil.copyfile(src, tmp)
    cap = cv2.VideoCapture(tmp, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
    ok, frame = cap.read()
    cap.release(); os.unlink(tmp)
    if not ok:
        raise RuntimeError("could not decode frame from mp4")
    return frame


# ───────────── high‑level latency harness ─────────────
def menu_latency_test(port:str, prepare_cmds:List[str],
                      open_cmd:str, close_cmd:str,
                      search_text:str, roi:Tuple[int,int,int,int],
                      threshold:int, iterations:int=1) -> Tuple[int,str]:
    """High‑level harness: open a menu, OCR for a keyword, measure latency.
    Robust to VideoRecordAgent hiccups: if the recorder thread fails to push
    a timestamp onto the queue we fall back to `datetime.now()` so the caller
    never blocks indefinitely.
    """
    lat = []
    for _ in range(iterations):
        flag  = f"{uuid.uuid4()}_{port}"
        video = rf"{VID_DIR}/record{port}/portview{port}.mp4"

        # prep: live + tune etc.
        for c in prepare_cmds:
            send_command(port, c)
            time.sleep(0.4)
        time.sleep(6)

        # start capture thread
        q = queue.Queue()
        t_thr = threading.Thread(target=_start_record, args=(flag, port, q), daemon=True)
        t_thr.start()
        time.sleep(2)

        # send the menu command we're timing
        t_before = datetime.now()
        send_command(port, open_cmd)
        time.sleep(2)
        if close_cmd:
            send_command(port, close_cmd)

        # stop recorder
        t_thr.join(timeout=10)
        _stop_record(flag)

        # pick timestamp: either from queue or fallback to now()
        if not q.empty():
            t0 = q.get()
        else:
            print("[WARN] VideoRecordAgent did not return start time; using fallback")
            t0 = datetime.now()

        delta = (t_before - t0).total_seconds() * 1000
        latency_ms = time_to_text(video, roi, search_text, 50, delta)
        lat.append(latency_ms)

        send_command(port, "LIVE")
        if latency_ms > threshold:
            break

    avg = round(sum(lat) / len(lat))
    return avg, ",".join(str(round(x, 2)) for x in lat)  # <- keep comma at end ",".join(str(round(x,2)) for x in lat)

# ───────────── result logging wrapper ─────────────
def classify_and_log(avg:int, thr:int, measurements:str,
                     uistephistoryid:str, portview:str):
    status = "false" if avg > thr else "true"
    subprocess.run([CLASSIFY_EXE, status, uistephistoryid])
    msg = (f"Menu exceeded {avg}\u202fms threshold."
           if status=="false" else f"Success. Avg={avg}\u202fms")
    evtype = "1" if status=="false" else "4"
    subprocess.run([EVENT_WRITER_EXE, "DPUnifiedMonitor",
                    evtype, "9000", "9001", msg, portview])
    sql = ("INSERT INTO uistephistory_more(idUIStepHistory,performancetime,measurements)"
           " VALUES (%s,%s,%s);")
    _mysql_insert(sql, (uistephistoryid, avg, measurements))


def _start_record(flag: str, port: str, q: queue.Queue) -> None:
    rec_dir = rf"{VID_DIR}/record{port}"
    os.makedirs(rec_dir, exist_ok=True)
    for f in os.listdir(rec_dir):
        file_path = os.path.join(rec_dir, f)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                print(f"[WARN] {file_path} locked, skipping delete")

    exe  = rf"{VID_DIR}/VideoRecordAgent.exe"
    name = rf"{rec_dir}/portview{port}.mp4"
    proc = subprocess.Popen([exe, flag, name], cwd=VID_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = proc.communicate()
    try:
        t0 = datetime.strptime(out.decode()[1:24], "%Y-%m-%d %H:%M:%S.%f")
    except (ValueError, IndexError):
        # recorder didn’t emit timestamp yet — just use now()
        t0 = datetime.now()
    q.put(t0)

def _stop_record(flag: str) -> None:
    marker = rf"{VID_DIR}/{flag}"
    if os.path.exists(marker):
        os.remove(marker)
    time.sleep(3)

def send_cmd(cmd: str, stb_id: str, **_):
    """
    Thin shim so dp_dispatcher can call dp_lib.send_cmd().
    Looks up <stb_id> in PORTVIEW_MAP; falls back to raw id.
    """
    port = reg.PORTVIEW_MAP.get(stb_id, stb_id)
    return send_command(port, cmd)
