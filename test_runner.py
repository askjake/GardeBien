"""test_runner.py
====================================
Declarative **test specification runner** for set‑top automated tests.
Specs are simple YAML files that list remote‑control *steps* and optional
expectations. The runner:

1. Parses the spec
2. Sends each command via `dp_dispatcher.send()` (so BF/AF + dataset
   capture happen automatically)
3. Uses `idm_model.predict()` to verify that the observed transition
   matches the *expected* command (if provided)
4. Records per‑step outcome and aggregates a final PASS/FAIL
5. Emits an in‑memory results dict which `reporting.py` can turn into
   JUnit / Allure artifacts.

YAML schema
-----------
```yaml
stb_id: "A"            # registered with dp_dispatcher
name: "dvr_create_timer"
steps:
  - cmd: "MENU"        # REQUIRED: command to send
    wait: 1.0           # OPTIONAL: seconds to sleep after send (default 0.5)
    expect_cmd: "MENU" # OPTIONAL: IDM‑predicted cmd we expect; omit to skip
    note: "open menu"  # OPTIONAL: free text

  - cmd: "RIGHT"
    expect_cmd: "RIGHT"

  - cmd: "OK"
```

Dependencies: `pyyaml`  (add to requirements.txt)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml, time, json, pathlib, logging
import reporting
from dp_dispatcher import send
from idm_model      import predict          # for “expect_cmd” checks
from ui_graph       import UIGraph

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

StepResult = Dict[str, Any]
RunResult = Dict[str, Any]

# ---------------------------------------------------------------------------
# Runner implementation
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_pair_paths() -> tuple[str, str]:
    """Return bf.jpg/af.jpg of the most recently saved dataset folder."""
    ds_root = Path("datasets")
    subdirs = [d for d in ds_root.iterdir() if d.is_dir()]
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    return str(latest / "bf.jpg"), str(latest / "af.jpg")


# ---------------------------------------------------------------------------
# Public runner API
# ---------------------------------------------------------------------------

def run_spec(spec_path: str | pathlib.Path, *, reporter=reporting):
    """
    Execute a YAML test-spec and return the run-record dict.
    """
    spec_path = pathlib.Path(spec_path)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    stb  = spec["stb_id"]
    name = spec.get("name", spec_path.stem)
    steps_out = []
    start_ts = time.time()
    
    if "capture_device" in spec and "portview" in spec:
        from stb_registry  import register_capture, register_portview
        from dp_dispatcher import register_stb
        register_capture(stb, spec["capture_device"])
        register_portview(stb, spec["portview"])
        register_stb(stb,
                 capture=spec["capture_device"],
                 portview=spec["portview"])

    for step in spec["steps"]:
        cmd   = step["cmd"].upper()
        note  = step.get("note", "")
        expect = step.get("expect_cmd", cmd).upper()
        wait   = float(step.get("wait", 0.5))
        repeat = int(step.get("repeat", 1))

        for i in range(repeat):
            send(cmd, stb_id=stb)     # send the key and capture BF/AF
            time.sleep(wait)          # respect the step’s wait

            bf, af = _latest_pair_paths()
            pred    = predict(bf, af)
            outcome = "PASS" if pred == expect else "FAIL"

            steps_out.append({
                "cmd": cmd,
                "expect_cmd": expect,
                "predicted_cmd": pred,
                "outcome": outcome,
                "timestamp": _utc_iso(),
                "note": step.get("note", ""),
                "iteration": i + 1,
            })

    run_rec = {
        "name": name,
        "stb_id": stb,
        "start_ts": start_ts,
        "end_ts": time.time(),
        "overall": "PASS" if all(s["outcome"] == "PASS" for s in steps_out) else "FAIL",
        "steps": steps_out,
        "spec": str(spec_path),
    }

    reporter.generate_run_report(run_rec)
    pathlib.Path("last_run.json").write_text(json.dumps(run_rec, indent=2))
    return run_rec
# ---------------------------------------------------------------------------
# CLI quick‑test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os

    ap = argparse.ArgumentParser(description="Run a YAML test spec.")
    ap.add_argument("spec", help="Path to YAML spec file")
    ap.add_argument("--neo_pass", default="mango-metal-moral-bronze-prague-8964", help="Neo4j password (falls back to env)")
    args = ap.parse_args()

    if args.neo_pass:
        os.environ["NEO4J_PASS"] = args.neo_pass   # picked up by UIGraph

    res = run_spec(args.spec)
    print(json.dumps(res, indent=2))