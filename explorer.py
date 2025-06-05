"""explorer.py
====================================
A *minimal‑viable* exploration agent that learns a quick policy for
navigating the UI graph using the embeddings from **idm_model.py**.
It is intentionally *simple* for Sprint‑1: ε‑greedy count‑based
exploration with a tabular value heuristic.

Key public functions
--------------------
* `Explorer(...)`  – agent class; call `act()` with a live frame to get
a command suggestion.
* `self_explore(budget_steps)` – runs an autonomous loop:
  1. grabs current frame via *capture device snapshot* (you’ll need to
     pass a function for that)
  2. chooses command
  3. dispatches via `dp_dispatcher.send`
  4. logs new edges into Neo4j.

Dependencies – nothing beyond the modules we’ve written plus numpy &
PyTorch (already installed for the IDM).

Feel free to replace this with an RL algorithm later; the rest of the
stack won’t need to change.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from idm_model import embed, predict
from ui_graph import UIGraph
from dp_dispatcher import send as dp_send

# All available actions (update if you add more commands)
_ACTION_SPACE: List[str] = [
    "UP","DOWN","LEFT","RIGHT","ENTER","MENU",
    "CH_DOWN","CH_UP","GUIDE","DVR","OPTIONS",
    "FFWD","RWD","PLAY","HOME"
]

# ---------------------------------------------------------------------------
# Simple count‑based ε‑greedy policy
# ---------------------------------------------------------------------------

class Policy:
    def __init__(self, actions: List[str], penalty: int = 10, bonus: int = 3):
        self.actions = actions
        # key = (state_hash, action)  value = visit count
        self.visits  = defaultdict(int)   # how many times we tried
        self.success = defaultdict(int)   # how many times screen changed
        self.penalty = penalty
        self.bonus   = bonus

    def choose(self, state_hash: str, epsilon: float = 0.4) -> str:
        if random.random() < epsilon:
            return random.choice(self.actions)
        scored = []
        for a in self.actions:
            key  = (state_hash, a)
            v    = self.visits[key]
            s    = self.success[key]
            score = v - self.bonus * s
            scored.append((score, a))
        scored.sort()
        return scored[0][1]
        
    def update(self, state_hash: str, action: str, changed: bool):
        """
        Increment visit-count. If `changed`==False, add extra penalty.
        """
        self.visits[(state_hash, action)] += 1
        if changed:
            self.success[(state_hash, action)] += 1
        else:
            # further discourage no-ops
            self.visits[(state_hash, action)] += self.penalty

# ---------------------------------------------------------------------------
# Explorer agent
# ---------------------------------------------------------------------------

class Explorer:
    def __init__(
        self,
        stb_id: str,
        grab_frame_fn: Callable[[], str],  # returns path to temp img
        ui_graph: UIGraph,
        epsilon: float = 0.2,
    ) -> None:
        """Parameters
        ----------
        stb_id : str
            Logical STB identifier (must be registered with dp_dispatcher).
        grab_frame_fn : Callable
            Function that captures a *single* current frame and returns
            the image **path** (JPEG/PNG) for embedding.
        ui_graph : UIGraph
            Open Neo4j wrapper instance to persist transitions.
        epsilon : float
            ε‑greedy temperature for exploration.
        """
        self.stb_id = stb_id
        self.grab_frame = grab_frame_fn
        self.ui_graph = ui_graph
        self.policy = Policy(_ACTION_SPACE)
        self.epsilon = epsilon

    # -------------------------------------------------------------
    # Main decision step
    # -------------------------------------------------------------


    def act(self) -> str:
        cur_img  = self.grab_frame()
        cur_hash = self._sha1(cur_img)

        cmd = self.policy.choose(cur_hash, self.epsilon)

        # 1) send & record BF/AF
        meta = dp_send(cmd, self.stb_id)
        time.sleep(2)   # let the UI settle

        # 2) grab AFTER frame and hash
        af_img  = self.grab_frame()
        af_hash = self._sha1(af_img)

        # 3) persist transition
        self.ui_graph.add_transition(cur_hash, af_hash, cmd, meta={})

        # 4) update policy, penalizing no-ops
        changed = (af_hash != cur_hash)
        self.policy.update(cur_hash, cmd, changed)

        return cmd

    # -------------------------------------------------------------

    def self_explore(self, budget_steps: int = 10000):
        for step in range(1, budget_steps + 1):
            cmd = self.act()
            print(f"[explore] step {step}/{budget_steps} – sent {cmd}")

    # -------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------

    @staticmethod
    def _sha1(img_path: str) -> str:
        import hashlib
        h = hashlib.sha1()
        with open(img_path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

# ---------------------------------------------------------------------------
# grab_frame_fn using ffmpeg screenshots
# ---------------------------------------------------------------------------



import subprocess, tempfile, shutil, os, logging

log = logging.getLogger(__name__)

def _ffmpeg_screenshot(dshow_name: str, timeout: int = 5) -> str:
    """
    Grab one frame from a DirectShow device by (friendly) name.

    • Accepts both **raw names**  ("Video (02-0 Pro Capture Quad HDMI)")
      and strings that already start with "video=".
    • Always passes a single well‑formed `-i video=NAME` to ffmpeg.
    """
    import subprocess, tempfile, shutil, os, logging, re
    log = logging.getLogger(__name__)

    # -------- normalise device string --------
    if dshow_name.lower().startswith("video="):
        dev_arg = dshow_name                     # already good
    else:
        # Escape leading/trailing quotes if user copied them
        dev_arg = "video=" + re.sub(r'^"|"$', "", dshow_name)

    # -------- build temp file --------
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    cmd = [
        "ffmpeg",
        "-y", "-loglevel", "error",
        "-f", "dshow",
        "-i", dev_arg,
        "-vframes", "1",
        "-q:v", "2",
        tmp_path,
    ]

    try:
        subprocess.run(cmd, timeout=timeout,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE,
                       check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        stderr = getattr(exc, "stderr", b"").decode(errors="ignore")
        raise RuntimeError(f"ffmpeg screenshot failed for '{dev_arg}':\n{stderr}") from exc

    if os.path.getsize(tmp_path) == 0:
        os.unlink(tmp_path)
        raise RuntimeError("ffmpeg wrote zero‑byte screenshot")

    log.debug("Captured screenshot → %s", tmp_path)
    return tmp_path

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Explorer quick demo")
    ap.add_argument("stb_id",   help="Logical STB identifier (e.g. A)")
    ap.add_argument("portview", help="DP-Studio portview number (e.g. 8)")
    ap.add_argument("device",   help="ffmpeg device string (e.g. video=...)")
    ap.add_argument("--steps",   type=int,   default=10_000,
                    help="Number of exploration steps")
    ap.add_argument("--epsilon", type=float, default=0.2,
                    help="ε-greedy exploration rate (0 → greedy, 1 → random)")
    args = ap.parse_args()

    # 1) register the STB
    from dp_dispatcher import register_stb
    register_stb(
        args.stb_id,
        capture=args.device,
        portview=args.portview
    )

    # 2) build grab_frame fn
    def grab():
        return _ffmpeg_screenshot(args.device)

    # 3) open Neo4j & launch explorer with custom epsilon
    from ui_graph import UIGraph
    with UIGraph() as g:
        exp = Explorer(
            stb_id=args.stb_id,
            grab_frame_fn=grab,
            ui_graph=g,
            epsilon=args.epsilon        # ← use the CLI value here
        )
        exp.self_explore(args.steps)
