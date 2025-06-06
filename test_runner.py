#!/usr/bin/env python3
"""
Declarative test-spec runner for GardeBien.

YAML schema
-----------
stb_id: "1"
name:   "guide_navigation1"
steps:
  - cmd: "MENU"
    wait: 1.0
    expect_cmd: "MENU"
  - cmd: "RIGHT"
    expect_cmd: "RIGHT"
"""

from __future__ import annotations
import json, time, logging, pathlib, yaml
from datetime import datetime, timezone
from typing import Any, Dict, List

from dp_dispatcher        import send
from capture_and_detector import capture_frame, detect_screen   # single-frame path
from idm_model            import predict                       # pair BF/AF path
import reporting

# ───────────────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# ---------------------------------------------------------------------------
# Types
StepResult = Dict[str, Any]
RunResult  = Dict[str, Any]

# ---------------------------------------------------------------------------
def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_pair_paths() -> tuple[str, str] | None:
    """Return (bf.jpg, af.jpg) of **newest** dataset folder, else None."""
    ds_root = pathlib.Path("datasets")
    jpgs    = sorted(ds_root.rglob("bf.jpg"), key=lambda p: p.stat().st_mtime)
    if not jpgs:
        return None
    bf = jpgs[-1]
    af = bf.with_name("af.jpg")
    return str(bf), str(af) if af.exists() else None


# ---------------------------------------------------------------------------
def run_spec(spec_path: str | pathlib.Path) -> RunResult:
    """
    Execute a YAML test-spec and return an in-memory results dict.
    """
    spec_path = pathlib.Path(spec_path)
    spec      = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    if "capture_device" in spec and "portview" in spec:
        from stb_registry    import register_capture, register_portview
        from dp_dispatcher   import register_stb

        cap = spec["capture_device"]
        pv  = spec["portview"]
        stb = spec["stb_id"]
        register_capture( stb, cap )
        register_portview( stb, pv )
        register_stb( stb, capture=cap, portview=pv )
        log.info(f"Registered STB {stb} → {cap} (portview={pv})")
         
    stb_id   = spec["stb_id"]
    run_name = spec.get("name", spec_path.stem)
    steps_out: List[StepResult] = []

    log.info(f"▶️  RUN: {run_name} on STB={stb_id}")

    start_ts = time.time()

    for step_idx, step in enumerate(spec["steps"], 1):
        cmd     = step["cmd"].upper()
        expect  = step.get("expect_cmd", cmd).upper()
        wait    = float(step.get("wait", 0.5))
        repeat  = int(step.get("repeat", 1))
        note    = step.get("note", "")

        for rep in range(repeat):
            log.info(f"  [{step_idx}.{rep+1}] send {cmd}")
            send(cmd, stb_id=stb_id)
            time.sleep(wait)

            # ---- prediction ------------------------------------------------
            bf_af = _latest_pair_paths()
            if bf_af:
                bf, af = bf_af
                pred = predict(bf, af)                 # pair-based check
                conf = 1.0                             # dummy (pair path returns str)
            else:
                frame = capture_frame()                # grab HDMI now
                pred, conf = detect_screen(frame)      # single frame
            # -----------------------------------------------------------------

            outcome = "PASS" if pred == expect else "FAIL"
            log.info(f"      → {pred} ({conf:.2f})  [{outcome}]")

            steps_out.append({
                "step_idx":    step_idx,
                "iteration":   rep + 1,
                "cmd":         cmd,
                "expect_cmd":  expect,
                "predicted":   pred,
                "confidence":  conf,
                "outcome":     outcome,
                "note":        note,
                "timestamp":   _utc_iso(),
            })

    run_rec: RunResult = {
        "name":     run_name,
        "stb_id":   stb_id,
        "start_ts": start_ts,
        "end_ts":   time.time(),
        "overall":  "PASS" if all(s["outcome"] == "PASS" for s in steps_out) else "FAIL",
        "steps":    steps_out,
        "spec":     str(spec_path),
    }

    reporting.generate_run_report(run_rec)          # JUnit / Allure files
    pathlib.Path("last_run.json").write_text(json.dumps(run_rec, indent=2))
    log.info(f"🏁  RUN COMPLETE — {run_rec['overall']}")
    return run_rec


# ---------------------------------------------------------------------------
# CLI
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("test", help="YAML path, e.g. tests\\guide_navigation1.yaml")
    args = ap.parse_args()

    ok = run_spec(args.test)
    sys.exit(0 if ok["overall"] == "PASS" else 1)
