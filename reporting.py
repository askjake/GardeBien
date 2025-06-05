"""reporting.py
====================================
Turns a **RunResult** dict (output of `test_runner.run_spec`) into
JUnit‑style XML (compatible with Jenkins, GitLab, etc.) and drops it in
`reports/`.  Optionally emits a lightweight Allure JSON record as well –
useful if you later wire Allure into your CI.

Public helpers
--------------
* `generate_run_report(run_result, junit=True, allure=False)` – returns
  a dict of generated file paths.

Dependencies: std‑lib only (`xml.etree.ElementTree`, `json`, `pathlib`).
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------

_REPORT_DIR = Path("reports")
_REPORT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helper – JUnit serialization
# ---------------------------------------------------------------------------

def _to_junit_xml(run: Dict) -> str:
    ts_iso = datetime.now(timezone.utc).isoformat()
    suite = ET.Element("testsuite", {
        "name": run["name"],
        "tests": str(len(run["steps"])),
        "failures": str(sum(1 for s in run["steps"] if s["outcome"] == "FAIL")),
        "timestamp": ts_iso,
    })

    for step in run["steps"]:
        tc = ET.SubElement(suite, "testcase", {
            "classname": run["name"],
            "name": step["cmd"],
            "time": "0",
        })
        if step["outcome"] == "FAIL":
            msg = f"expected={step['expect_cmd']} predicted={step['predicted_cmd']}"
            ET.SubElement(tc, "failure", {"message": msg}).text = msg
        # Optional system‑out JSON for rich logging
        ET.SubElement(tc, "system-out").text = json.dumps(step, indent=2)

    xml_str = ET.tostring(suite, encoding="utf-8").decode()
    return xml_str

# ---------------------------------------------------------------------------
# Helper – Allure JSON (very trimmed)
# ---------------------------------------------------------------------------

def _to_allure_json(run: Dict) -> str:
    allure_run = {
        "name": run["name"],
        "status": run["overall"].lower(),
        "steps": [
            {
                "name": s["cmd"],
                "status": s["outcome"].lower(),
                "attachments": [],
            }
            for s in run["steps"]
        ],
        "start": run["start_ts"],
        "stop": run["end_ts"],
    }
    return json.dumps(allure_run, indent=2)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_run_report(
    run_result: Dict,
    junit: bool = True,
    allure: bool = False,
) -> Dict[str, Path]:
    """Generate XML/JSON files. Returns dict with file paths."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name_slug = run_result["name"].replace(" ", "_")

    outputs: Dict[str, Path] = {}

    if junit:
        xml_path = _REPORT_DIR / f"{name_slug}_{ts}.xml"
        xml_path.write_text(_to_junit_xml(run_result), encoding="utf-8")
        outputs["junit_xml"] = xml_path

    if allure:
        json_path = _REPORT_DIR / f"{name_slug}_{ts}.json"
        json_path.write_text(_to_allure_json(run_result), encoding="utf-8")
        outputs["allure_json"] = json_path

    return outputs

# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple synthetic run result for smoke test
    sample = {
        "name": "demo_run",
        "start_ts": datetime.now(timezone.utc).isoformat(),
        "end_ts": datetime.now(timezone.utc).isoformat(),
        "overall": "PASS",
        "steps": [
            {"cmd": "RIGHT", "expect_cmd": "RIGHT", "predicted_cmd": "RIGHT", "outcome": "PASS"},
            {"cmd": "OK", "expect_cmd": "OK", "predicted_cmd": "LEFT", "outcome": "FAIL"},
        ],
    }
    outs = generate_run_report(sample, allure=True)
    print("Generated:", outs)
