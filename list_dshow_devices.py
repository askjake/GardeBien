# list_dshow_devices.py  (run inside your venv)
import json, subprocess, re, sys, shutil, pathlib

ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
cmd = [ffmpeg, "-list_devices", "true", "-f", "dshow", "-i", "dummy"]

print(">>> probing DirectShow devices …\n")
result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, check=False)

devices = re.findall(r'\[dshow @ .*?\] +"(?P<name>[^"]+)"', result.stderr)
device_map = {i + 1: name for i, name in enumerate(devices) if "video" in name.lower()}

for idx, name in device_map.items():
    print(f"{idx:>2}.  {name}")

out = pathlib.Path("capture_devices.json")
out.write_text(json.dumps(device_map, indent=2))
print("\nSaved →", out)
