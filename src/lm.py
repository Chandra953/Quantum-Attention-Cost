"""
Character-level language modelling on WikiText-2 (real text, beyond synthetic recall).

Data: the WikiText-2 release used by the PyTorch word_language_model example
(https://github.com/pytorch/examples/tree/main/word_language_model/data/wikitext-2),
modelled at the character level. Download with scripts/get_wikitext2.sh into data/wikitext-2/.

Models: 4-layer causal pre-LN Transformers, d = 128, 4 heads, context 256 characters.
Layers alternate a sliding-window softmax layer (window 64) with a global layer whose
attention is the variant under test, the same hybrid design as the recall experiment:
    window-w64 + X + window-w64 + X,   X in {softmax, cpla, taylor-d8, elu-d81, performer-d16, window-w64}
Metric: bits per character (BPC) on the held-out test split.

Usage: python src/lm.py "softmax,cpla" 3000 2      # kinds, training steps, seeds
"""
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recall import Block  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "wikitext-2")
OUT = os.path.join(HERE, "..", "results", "lm_wikitext2.csv")
torch.set_num_threads(max(1, os.cpu_count() or 1))


def load():
    text = {s: open(os.path.join(DATA, f"{s}.txt"), encoding="utf-8").read() for s in ("train", "valid", "test")}
    chars = sorted(set(text["train"]))
    idx = {c: i for i, c in enumerate(chars)}
    unk = len(chars)                                     # characters unseen in training
    enc = {s: torch.tensor([idx.get(c, unk) for c in t], dtype=torch.long) for s, t in text.items()}
    return enc, len(chars) + 1


class LM(nn.Module):
    def __init__(self, vocab, d, heads, kinds, ctx):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(0.02 * torch.randn(ctx, d))
        self.blocks = nn.ModuleList(Block(d, heads, k) for k in kinds)
        for b in self.blocks:
            b.attn.use_features = True       # CPLA via its exact 3^n feature map
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.emb(ids) + self.pos[: ids.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def batches(data, B, ctx, g):
    starts = torch.randint(0, len(data) - ctx - 1, (B,), generator=g)
    x = torch.stack([data[s:s + ctx] for s in starts])
    y = torch.stack([data[s + 1:s + ctx + 1] for s in starts])
    return x, y


@torch.no_grad()
def evaluate(model, data, ctx, max_windows=None):
    """Mean BPC over non-overlapping windows of the split (every character predicted once)."""
    model.eval()
    n = (len(data) - 1) // ctx
    if max_windows:
        n = min(n, max_windows)
    total, count = 0.0, 0
    for i in range(0, n, 32):
        rows = range(i, min(i + 32, n))
        x = torch.stack([data[r * ctx:(r + 1) * ctx] for r in rows])
        y = torch.stack([data[r * ctx + 1:(r + 1) * ctx + 1] for r in rows])
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum")
        total += loss.item(); count += y.numel()
    model.train()
    return total / count / math.log(2)


def run(kind, seed, enc, vocab, steps=3000, B=32, ctx=256, d=128, heads=4, lr=2e-3):
    kinds = ["window-w64", kind, "window-w64", kind]
    torch.manual_seed(seed)
    model = LM(vocab, d, heads, kinds, ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(1000 + seed)
    t0 = time.time()
    for step in range(steps):
        x, y = batches(enc["train"], B, ctx, g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            print(f"  {kind} seed {seed} step {step} train_bpc {loss.item() / math.log(2):.3f}", flush=True)
    return dict(model=kind, seed=seed, steps=steps, valid_bpc=evaluate(model, enc["valid"], ctx),
                test_bpc=evaluate(model, enc["test"], ctx),
                params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0))


if __name__ == "__main__":
    kinds = sys.argv[1].split(",")
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    seeds = range(int(sys.argv[3])) if len(sys.argv) > 3 else range(2)
    enc, vocab = load()
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {(r["model"], int(r["seed"]), int(r["steps"])) for r in csv.DictReader(fh)}
    for s in seeds:
        for kind in kinds:
            if (kind, s, steps) in done:
                continue
            r = run(kind, s, enc, vocab, steps=steps)
            print(r, flush=True)
            new = not os.path.exists(OUT)
            with open(OUT, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(r.keys()))
                if new:
                    w.writeheader()
                w.writerow(r)
