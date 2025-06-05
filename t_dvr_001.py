#!/usr/bin/env python3
"""
T_DVR_001  –  build a custom timer, wait for it to record, play it back,
then stop the recording.

Uses visual comparison (SSIM) instead of OCR.  Needs:
  • ui_map.csv      (with   before_img, before_img_crop_rel,
                      thresh_pre, after_img, after_img_crop_rel,
                      thresh_post, Breadcrumb  columns)
  • screenshots/…   referenced therein
  • dp_lib.py       (the capture‑card helper you tuned earlier)
"""

from __future__ import annotations
import time
import subprocess, sys, pathlib
import cv2, pandas as pd, numpy as np
import dp_lib as gng
from dp_dispatcher import register_stb, send

# ───────────── Remote-command map ─────────────
PORT_MASK_CMD = {
    "CMD_DVR":     "DVR",
    "CMD_UP":      "UP",
    "CMD_DOWN":    "DOWN",
    "CMD_LEFT":    "LEFT",
    "CMD_RIGHT":   "RIGHT",
    "CMD_SELECT":  "SELECT",
    "CMD_OPTIONS": "OPTIONS",
    "CMD_LIVE_TV": "LIVE",
    "CMD_RESET_USER_SETTINGS": "RESET_USER_SETTINGS",
}

# ───────────── Load ui_map.csv ─────────────
SCRIPT_DIR = pathlib.Path(__file__).parent
DF = pd.read_csv(SCRIPT_DIR / "ui_map.csv")

# find threshold columns
thresh_cols = [c for c in DF.columns if c.lower().startswith("thresh")]
if len(thresh_cols) == 1:
    DF["thresh_pre"]  = DF[thresh_cols[0]].astype(float)
    DF["thresh_post"] = DF["thresh_pre"]
else:
    DF["thresh_pre"]  = DF[thresh_cols[0]].astype(float)
    DF["thresh_post"] = DF[thresh_cols[1]].astype(float)

UI_INFO: dict[str, dict] = {}
for _, r in DF.iterrows():
    ctx        = str(r["UI_Context"]).strip()
    breadcrumb = str(r.get("Breadcrumb","")).strip()

    def _path(col:str):
        raw = str(r.get(col,"")).strip()
        if not raw or raw.lower()=="nan": return None
        p = raw.replace("\\","/")
        c1 = SCRIPT_DIR / p
        c2 = SCRIPT_DIR / "screenshots" / pathlib.Path(p).name
        if c1.is_file(): return c1.as_posix()
        if c2.is_file(): return c2.as_posix()
        return None

    def _parse_crop(col:str):
        raw = str(r.get(col,"")).strip()
        if not raw or raw.lower()=="nan": return None
        nums = [float(x) for x in raw.split(",")]
        return tuple(nums) if len(nums)==4 else None

    UI_INFO[ctx] = {
        "breadcrumb": breadcrumb,
        "pre_img":   _path("before_img"),
        "pre_crop":  _parse_crop("before_img_crop_rel"),
        "pre_thr":   float(r["thresh_pre"]),
        "post_img":  _path("after_img"),
        "post_crop": _parse_crop("after_img_crop_rel"),
        "post_thr":  float(r["thresh_post"]),
    }

# ───────────── Image helpers ─────────────
def _crop(frame: np.ndarray, rel:tuple[float,float,float,float]|None) -> np.ndarray:
    if not rel:
        return frame
    h,w = frame.shape[:2]
    a,b,c,d = rel
    if any(v>1 for v in rel):
        xs = sorted((a,c)); ys = sorted((b,d))
        x1,x2 = map(int, xs); y1,y2 = map(int, ys)
    else:
        x1,y1 = int(a*w), int(b*h)
        x2,y2 = int(c*w), int(d*h)
    x1,x2 = max(0,x1), min(w,x2)
    y1,y2 = max(0,y1), min(h,y2)
    return frame[y1:y2, x1:x2]

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return ssim(ga, gb)

# ───────────── High-level assert ─────────────
def assert_screen(mask:int, ctx:str, cmd:str|None) -> tuple[float,float]:
    info = UI_INFO[ctx]
    live0 = gng.grab_current_frame(mask)

    def _compare(live: np.ndarray,
                 ref_path: str | None,
                 thr: float,
                 tag: str,
                 crop_rel: tuple[float, float, float, float] | None
                 ) -> tuple[float, bool]:
        if not (ref_path and pathlib.Path(ref_path).exists()):
            print(f"[WARN] no ref for {ctx} {tag}")
            return 0.0, True

        ref_img = cv2.imread(ref_path)
        live_c  = _crop(live,  crop_rel)
        ref_c   = _crop(ref_img, crop_rel)

        if live_c.size == 0 or ref_c.size == 0:
            cv2.imwrite("debug_fail_full.png", live0)
            raise AssertionError(f"{tag} {ctx}: empty crop {crop_rel}")

        sim = _ssim(live_c, ref_c)
        if sim >= thr:
            return sim, True

        # ───── failure handling / preview ─────
        cv2.imwrite("debug_fail_full.png", live)
        cv2.imwrite("debug_fail_crop.png",  live_c)
        cv2.imwrite("debug_ref_crop.png",   ref_c)
        print(f"[FAIL] {tag} {ctx}: sim={sim:.3f}  (thr={thr})")
        print("        press  T  in the preview window to tune crop")

        # draw full‑frame outlines in green
        ref_vis  = ref_img.copy()
        live_vis = live.copy()
        l,t,r,b  = crop_rel or (0,0,1,1)
        for vis, img in ((ref_vis, ref_img), (live_vis, live)):
            h, w = img.shape[:2]
            cv2.rectangle(vis,
                          (int(l*w), int(t*h)),
                          (int(r*w), int(b*h)),
                          (0,255,0), 2)

        # horizontally stack at same height
        h_ref,_  = ref_vis.shape[:2]
        h_live,_ = live_vis.shape[:2]
        target_h = max(h_ref, h_live)
        def _resize_to_height(im):
            h0, w0 = im.shape[:2]
            scale  = target_h / h0
            return cv2.resize(im, (int(w0*scale), target_h))

        preview = np.hstack((_resize_to_height(ref_vis),
                             _resize_to_height(live_vis)))
        cv2.imshow("Validation fail  (REF | LIVE)", preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key in (ord('t'), ord('T')):
            from crop_tuner import tune
            which = ("after_img_crop_rel" if tag=="post"
                     else "before_img_crop_rel")
            new_rel = tune(ref_path, live, crop_rel, ctx,
                           which_field=which)
            if new_rel:
                # store back as a tuple
                if tag=="post":
                    info["post_crop"] = tuple(new_rel)
                else:
                    info["pre_crop"]  = tuple(new_rel)
                print("[INFO] crop updated – re‑checking …")
                return _compare(live, ref_path, thr, tag, new_rel)

        return sim, False

    # ─ pre ─
    pre_sim, pre_ok = _compare(
        gng.grab_current_frame(mask),
        info["pre_img"], info["pre_thr"], "pre", info["pre_crop"]
    )
    if not pre_ok:
        raise AssertionError(f"pre‑check failed for {ctx}")

    # send key
    if cmd:
        port = str(gng.mask_to_single_port(mask))
        gng.send_command(port, cmd if cmd.isdigit()
                                else PORT_MASK_CMD[cmd])
        time.sleep(0.8)

    # ─ post ─
    post_sim, post_ok = _compare(
        gng.grab_current_frame(mask),
        info["post_img"], info["post_thr"], "post", info["post_crop"]
    )
    return pre_sim, post_sim, post_ok

# ───────────── Test flow ─────────────
STEP_NUM = 1
# print header
print(f"{'Step':>4} | {'Breadcrumb':40} | {'Pre':>6} | {'Cmd':11} | {'P‑Sim':>8} | {'Thr':>5} ")
print("-"*100)

def step(mask: int, ctx: str, code: str | None = None, sleep: float = 0.0):
    global STEP_NUM
    info = UI_INFO[ctx]
    try:
        pre_s, post_s, ok = assert_screen(mask, ctx, code)
    except AssertionError as e:
        # catastrophic issue (empty crop, etc.) → re‑raise after note
        print(f"\n[ABORT] {e}\n"); raise

    # print row (even if similarity failed)
    bc  = (info["breadcrumb"] or ctx)[:40]
    thr = info["post_thr"]
    cmd = code or ""
    flag = "" if ok else " ✗"
    print(f"{STEP_NUM:4d} | {bc:40s} | {pre_s:6.3f} | {cmd:11s} | {post_s:6.3f}{flag} | {thr:4.2f}")

    STEP_NUM += 1
    if not ok:                               # stop test on failure
        raise AssertionError(f"Step {STEP_NUM-1} failed ({ctx})")
    if sleep:
        time.sleep(sleep)


def run(mask:int):
    # 0. reset STB user settings
    portview = gng.mask_to_single_port(mask)
    subprocess.run([
        sys.executable,
        str(SCRIPT_DIR / "reset_stb_user_settings.py"),
        portview
    ], check=True)
    time.sleep(1)  # allow reset to process

    step(mask, "DVR Menu (root)",        "CMD_DVR",  1.4)
    step(mask, "DVR – Recordings Tab",   "CMD_UP")
    step(mask, "DVR – Schedule Tab",     "CMD_RIGHT")
    step(mask, "DVR – Timers Tab",       "CMD_RIGHT")
    step(mask, "Timers Options",         "CMD_OPTIONS")
    step(mask, "Create Custom Timer (blank)", "3")
    # 2. choose channel 2
    step(mask, "Select Channel list", "CMD_SELECT")
    step(mask, "KEY_2", "2",0.5)
    step(mask, "Channel 2 listed", "CMD_SELECT")          # 9

    # 3. Frequency ▸ Date ▸ Time
    step(mask, "Custom Timer – Frequency field", "CMD_DOWN")
    step(mask, "Custom Timer – Date field",      "CMD_DOWN")
    step(mask, "Custom Timer – Time field",      "CMD_DOWN")

    # 4. open set-time dialog + tweak
    step(mask, "Set-Time dialog (hours)", "CMD_SELECT")
    for cmd in ("DOWN","DOWN","RIGHT"):
        gng.send_command(str(gng.mask_to_single_port(mask)), cmd)
    for _ in range(60):
        gng.send_command(str(gng.mask_to_single_port(mask)), "DOWN")
    for cmd in ("RIGHT","RIGHT","DOWN","RIGHT","UP","UP","RIGHT","RIGHT"):
        gng.send_command(str(gng.mask_to_single_port(mask)), cmd)
    time.sleep(3)
    step(mask, "Set-Time dialog (Save)", "CMD_SELECT", 2)
    step(mask, "Custom Timer – Time field")

    # 5. select 4 h ▸ Create
    step(mask, "Custom Timer “?” (4th pos)", "CMD_RIGHT")
    step(mask, "Custom Timer Create btn",    "CMD_RIGHT")
    step(mask, "Timers Tab (after create)",  "CMD_SELECT", 2)

    # 6. back to Live → wait 2 min
    step(mask, "Live AV", "CMD_LIVE_TV")
    time.sleep(120)  # wait for recording to start

    # 7. confirm active recording
    step(mask, "DVR Menu (active recording)", "CMD_DVR", 1)
    gng.send_command(str(gng.mask_to_single_port(mask)), "CMD_DOWN")
    gng.send_command(str(gng.mask_to_single_port(mask)), "CMD_LEFT")
    step(mask, "Program Info (recording)",    "CMD_SELECT",1)
    step(mask, "Playback banner (recording)", "CMD_SELECT",1)  # 22

    # 8. stop
    gng.send_command(str(gng.mask_to_single_port(mask)), "SELECT")
    gng.send_command(str(gng.mask_to_single_port(mask)), "SELECT")
    gng.send_command(str(gng.mask_to_single_port(mask)), "LEFT")
    step(mask, "Popup #312 Stop", "CMD_SELECT")
    step(mask, "Live AV",           "CMD_SELECT")


# ───────────── CLI entrypoint ─────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: t_dvr_001.py <PortMask>")
        sys.exit(1)
    run(int(sys.argv[1]))
    sys.exit(0)
