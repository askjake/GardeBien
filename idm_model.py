##!/usr/bin/env python3
"""
idm_model.py
============
Vision *Inverse-Dynamics Model* (IDM)

Now supports multi-GPU training, mixed precision, and optimized data loading.

Usage:

    # train from scratch or fine-tune
    python idm_model.py train --epochs 25 --batch_size 16 --val_split 0.1

    # predict a single step
    python idm_model.py predict bf.jpg af.jpg
"""
import itertools
import json
import random
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate
from torchvision.io import read_video
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset_store import iter_batches

# ─── Config & Paths ────────────────────────────────────────────────────────
DATASET_ROOT = Path(os.getenv("GB_DATASETS", "datasets")).expanduser()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
WEIGHTS_PATH = Path(os.getenv("GB_MODELS", "models")).expanduser() / "idm_resnet.pt"
WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── Utilities ─────────────────────────────────────────────────────────────

def _load_img(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img.transpose(2, 0, 1) - 0.5) / 0.5
    return torch.from_numpy(img)

def video_collate(batch):
    videos, bfs, afs, ys = zip(*batch)
    bf_batch = default_collate(bfs)
    af_batch = default_collate(afs)
    y_batch = default_collate(ys)
    return list(videos), bf_batch, af_batch, y_batch

# ─── Dataset ───────────────────────────────────────────────────────────────
class ClipBF_AFDataset(IterableDataset):
    def __init__(self, samples: List[Tuple[Path,Path,Path,str]], cmd2idx: Dict[str,int], val_split: float, train: bool):
        self.samples = samples
        self.cmd2idx = cmd2idx
        self.val_split = val_split
        self.train = train

    def __iter__(self):
        n = len(self.samples)
        split_idx = int((1 - self.val_split) * n)
        for idx, (clip, bf_p, af_p, cmd) in enumerate(self.samples):
            is_train = idx < split_idx
            if (self.train and is_train) or (not self.train and not is_train):
                vid = None
                bf = _load_img(Path(bf_p))
                af = _load_img(Path(af_p))
                label = self.cmd2idx[cmd]
                yield vid, bf, af, label

# ─── Model ─────────────────────────────────────────────────────────────────
class SiameseIDM(nn.Module):
    def __init__(self, num_actions:int, emb_dim:int = 128):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.trunk = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, emb_dim),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Linear(2*emb_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions)
        )

    def forward(self, bf, af):
        f1 = self.proj(self.trunk(bf))
        f2 = self.proj(self.trunk(af))
        x = torch.cat([f1, f2], dim=1)
        return self.head(x), f2

# ─── Training ──────────────────────────────────────────────────────────────
def train(dataset_dir: str | Path = DATASET_ROOT, epochs: int = 3, batch_size: int = 32, val_split: float = 0.1):
    # build vocab
    cmd_counts = defaultdict(int)
    for batch in iter_batches(dataset_dir, batch_size=128):
        for _, _, _, cmd in batch:
            cmd_counts[cmd] += 1
    cmd2idx = {c:i for i,c in enumerate(sorted(cmd_counts))}
    idx2cmd = {i:c for c,i in cmd2idx.items()}

    samples = list(itertools.chain(*iter_batches(dataset_dir, batch_size=128)))
    random.shuffle(samples)

    train_ds = ClipBF_AFDataset(samples, cmd2idx, val_split, train=True)
    val_ds = ClipBF_AFDataset(samples, cmd2idx, val_split, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() or 4,
        collate_fn=video_collate,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() or 4,
        collate_fn=video_collate,
        pin_memory=True
    )

    # model, parallelize if multiple GPUs
    model = SiameseIDM(len(cmd2idx)).to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("Loaded existing weights")

    criterion = nn.CrossEntropyLoss()
    optimizr = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        tot, corr, ls_sum = 0, 0, 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}")
        for _, bf, af, y in pbar:
            bf, af, y = bf.to(DEVICE), af.to(DEVICE), y.to(DEVICE)
            optimizr.zero_grad()
            with autocast():
                logits, _ = model(bf, af)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizr)
            scaler.update()

            preds = logits.argmax(1)
            corr += (preds == y).sum().item()
            tot += y.size(0)
            ls_sum += loss.item() * y.size(0)
            pbar.set_postfix(acc=corr / tot, loss=ls_sum / tot)

        # validation
        model.eval()
        v_tot, v_corr = 0, 0
        with torch.no_grad():
            for _, bf, af, y in val_loader:
                bf, af, y = bf.to(DEVICE), af.to(DEVICE), y.to(DEVICE)
                with autocast(): logits, _ = model(bf, af)
                v_corr += (logits.argmax(1) == y).sum().item()
                v_tot += y.size(0)
        print(f"Epoch {epoch} val_acc={v_corr/v_tot:.3f}")

        torch.save(model.state_dict(), WEIGHTS_PATH)
        (WEIGHTS_PATH.parent / "cmd_vocab.json").write_text(json.dumps(idx2cmd, indent=2))

    print("✅ Training complete — weights at", WEIGHTS_PATH)

# ─── Inference ──────────────────────────────────────────────────────────────
def _lazy_load():
    global _model_cache
    model, idx2cmd = _model_cache
    if model is None:
        # ─── load vocab & model ───────────────────────────────────────────
        raw = json.loads((WEIGHTS_PATH.parent / "cmd_vocab.json").read_text())
        idx2cmd = {int(k): v for k, v in raw.items()}   # ←★ cast keys to int
        model = SiameseIDM(len(idx2cmd)).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        model.eval()
        _model_cache = (model, idx2cmd)
    return model, idx2cmd


def _preprocess_nd(frame: np.ndarray) -> torch.Tensor:
    """BGR ndarray → 1×C×H×W float32 tensor on DEVICE, range [-1,1]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb.transpose(2, 0, 1) - 0.5) / 0.5
    return torch.from_numpy(rgb).unsqueeze(0).to(DEVICE)

def predict_single(frame: np.ndarray) -> Tuple[str, float]:
    """
    Return (label, confidence) for one frame.  We feed the SAME image into the
    Siamese heads twice – not perfect, but good enough for real-time routing.
    """
    model, idx2cmd = _lazy_load()
    batch = _preprocess_nd(frame)
    with torch.no_grad():
        logits, _ = model(batch, batch)          # bf = af = frame
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(probs.argmax())
    label = idx2cmd[str(idx)]
    conf  = float(probs[idx])
    return label, conf

# Export a stable API the rest of the pipeline can import
preprocess_ndarray = _preprocess_nd          # nice to have
predict_single_frame = predict_single        # idem

# Optional: expose LABELS if other code wants it
try:
    _, _idx2cmd = _lazy_load()
    LABELS = [_idx2cmd[str(i)] for i in range(len(_idx2cmd))]
except Exception:
    LABELS = []

def predict(bf_path: Path, af_path: Path) -> str:
    model, idx2cmd = _lazy_load()
    bf = _load_img(Path(bf_path)).unsqueeze(0).to(DEVICE)
    af = _load_img(Path(af_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(bf, af)
    return idx2cmd[logits.argmax(1).item()]

def embed(img_path: Path) -> torch.Tensor:
    model, _ = _lazy_load()
    img = _load_img(Path(img_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, emb = model(img, img)
    return emb.squeeze(0).cpu()

# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__=="__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="IDM train / predict")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train")
    p1.add_argument("--epochs",    type=int,   default=3)
    p1.add_argument("--batch_size",type=int,   default=32)
    p1.add_argument("--val_split", type=float, default=0.1)

    p2 = sub.add_parser("predict")
    p2.add_argument("bf")
    p2.add_argument("af")

    args = ap.parse_args()
    if args.cmd=="train":
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split
        )
    elif args.cmd=="predict":
        print(predict(Path(args.bf), Path(args.af)))
    else:
        sys.exit("unknown command")
