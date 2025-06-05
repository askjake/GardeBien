#!/usr/bin/env python3
"""
stb_sgs.py – send any SGS “back‑end” or “settings” command to a Dish STB.

Usage:
    python stb_sgs.py <portview> <command> [key=value]…

Examples:
  # start the remote‑pairing monitor
  python stb_sgs.py 4 start_remote_pairing_monitor

  # unpair a specific remote
  python stb_sgs.py 4 unpair_remote dev_id=1234

  # get Closed‐Caption settings
  python stb_sgs.py 4 get_stb_settings cid=1000 id=2 name=closed_caption

  # change Closed‐Caption font & service
  python stb_sgs.py 4 set_stb_settings cid=1000 id=2 name=closed_caption \
      data='{"service":1,"font_color":4}'
"""
import sys, subprocess, argparse
import dp_lib

# --- every supported command, per your list above ---
ALL_COMMANDS = [
    # back‑end
    "start_remote_pairing_monitor","stop_remote_pairing_monitor","get_paired_remote_list",
    "unpair_remote","locate_remote","get_remote_settings","set_remote_setting",
    "backup_devices","restore_device","query_stb_settings_from_remote", "reset_user_settings",
    "start_elcc_download",
    # get‑info
    "get_stb_information",
    # settings
    "set_stb_settings","get_stb_settings", "reset_stb_user_settings",
]

def main():
    p = argparse.ArgumentParser(
        description="Send an SGS command to a Dish STB via sgs_remote.py"
    )
    p.add_argument("portview",
                   help="PortView # as in UnifiedSettings.xml (1–16)")
    p.add_argument("command", choices=ALL_COMMANDS,
                   help="SGS command to invoke")
    p.add_argument("params", nargs="*",
                   help="key=value parameters, e.g. cid=1000 id=2 name=closed_caption data='…'")
    args = p.parse_args()

    # look up IP & STB‑ID from UnifiedSettings.xml
    ip  = dp_lib._xml_value(args.portview, "RESTAPIServerIP")
    stb = dp_lib._xml_value(args.portview, "RESTAPISTB")
    if not ip or not stb:
        print(f"❌ cannot find RESTAPI settings for port {args.portview}", file=sys.stderr)
        sys.exit(1)

    # build the sgs_remote.py invocation
    exe = "python.exe"
    wd  = f"{dp_lib.TOOLS_DIR}/DishSendRESTCommand"
    cmd = [exe,
           f"{wd}/sgs_remote.py",
           "-i", ip,
           "-s", stb,
           args.command]
    # append any key=value params
    cmd += args.params

    # run it
    proc = subprocess.run(cmd,
                          cwd=wd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True,
                          timeout=30)
    # print out whatever sgs_remote printed
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
