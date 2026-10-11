"""
Phase 1: associative recall with multi-layer Transformers.

Task (multi-query associative recall, in the style of the Zoology/MQAR benchmark):
    k1 v1 k2 v2 ... kK vK | q1 q2 ... qQ
Each query is one of the earlier keys; the model must output the value that
followed that key. Accuracy is measured on query positions only.

Models: 2-layer, multi-head, causal pre-LN Transformers that differ only in
the attention layer:
    softmax    : standard scaled dot-product attention          O(N^2)
    cp-softmax : softmax over the cosine-product (CP) kernel     O(N^2)
    cpla       : CP kernel, linear time, exact 3^n feature map   O(N)
    elu        : linear attention with elu+1 features            O(N)
    performer  : FAVOR+ positive random features                 O(N)

The CP kernel uses separate query and key angle projections:
    s_ij = prod_k cos^2((phi^Q_ik - phi^K_jk) / 2)
         = < f(phi^Q_i), f(phi^K_j) >,  f(phi) = ⊗_k [1, cos phi_k, sin phi_k] / sqrt(2)
which is exact with 3^n features (Proposition 1 of the paper, refactored).
"""
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------

def cp_features(phi):
    """Exact 3^n feature map of the cosine-product kernel. phi: (..., n) -> (..., 3^n)."""
    f = torch.stack([torch.ones_like(phi), torch.cos(phi), torch.sin(phi)], -1) / math.sqrt(2)
    out = f[..., 0, :]
    for k in range(1, phi.shape[-1]):
        out = (out[..., :, None] * f[..., k, None, :]).flatten(-2)
    return out


def causal_pairwise(phi_q, phi_k, v, eps=1e-6):
    """Same function as causal_linear, computed with an N x N masked kernel.
    Faster for the short training sequences used here; identical up to rounding."""
    N = phi_q.shape[2]
    s = (phi_q @ phi_k.transpose(-1, -2)) * torch.ones(N, N, device=v.device).tril()
    return (s @ v) / (s.sum(-1, keepdim=True) + eps)


LINEAR_IMPL = causal_pairwise        # set to causal_linear for the O(N) prefix-sum form


def causal_linear(phi_q, phi_k, v, eps=1e-6):
    """y_i = sum_{j<=i} (phi_q_i . phi_k_j) v_j / sum_{j<=i} phi_q_i . phi_k_j.
    Shapes (B, H, N, D) and (B, H, N, dv). Prefix sums give O(N D dv) time."""
    kv = torch.einsum("bhnd,bhne->bhnde", phi_k, v).cumsum(2)
    z = phi_k.cumsum(2)
    num = torch.einsum("bhnd,bhnde->bhne", phi_q, kv)
    den = (phi_q * z).sum(-1, keepdim=True) + eps
    return num / den


# ---------------------------------------------------------------------------
# Multi-head causal attention with interchangeable similarity
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, d, heads, kind, n_angles=4, dk=16, n_features=64):
        super().__init__()
        base, self.power, self.window = kind, 1, 4
        for part in kind.split("-")[1:]:
            if part[0] == "n" and part[1:].isdigit():
                n_angles = int(part[1:]); base = base.replace("-" + part, "")
            if part[0] == "p" and part[1:].isdigit():
                self.power = int(part[1:]); base = base.replace("-" + part, "")
            if part[0] == "d" and part[1:].isdigit():
                dk = int(part[1:]); base = base.replace("-" + part, "")
            if part[0] == "w" and part[1:].isdigit():
                self.window = int(part[1:]); base = base.replace("-" + part, "")
        kind = base
        self.h, self.kind, self.dv = heads, kind, d // heads
        if kind in ("cpla", "cp-softmax"):
            self.q = nn.Linear(d, heads * n_angles)
            self.k = nn.Linear(d, heads * n_angles)
            self.width = n_angles
            if kind == "cp-softmax":
                self.log_scale = nn.Parameter(torch.full((heads, 1, 1), math.log(5.0)))
        else:
            self.q = nn.Linear(d, heads * dk)
            self.k = nn.Linear(d, heads * dk)
            self.width = dk
        if kind == "performer":
            self.register_buffer("proj", torch.randn(heads, dk, n_features))
            self.m = n_features
        self.v, self.o = nn.Linear(d, d), nn.Linear(d, d)

    def split(self, x, w):
        B, N, _ = x.shape
        return x.view(B, N, self.h, w).transpose(1, 2)            # (B, H, N, w)

    def forward(self, x):
        B, N, _ = x.shape
        q, k = self.split(self.q(x), self.width), self.split(self.k(x), self.width)
        v = self.split(self.v(x), self.dv)
        causal = torch.ones(N, N, dtype=torch.bool, device=x.device).tril()

        if self.kind in ("softmax", "window"):
            s = q @ k.transpose(-1, -2) / math.sqrt(self.width)
            allowed = causal
            if self.kind == "window":     # sliding-window softmax: attend to the last w tokens only
                idx = torch.arange(N, device=x.device)
                allowed = causal & ((idx[:, None] - idx[None, :]) < self.window)
            y = F.softmax(s.masked_fill(~allowed, -1e9), -1) @ v
        elif self.kind == "taylor":       # Based: 2nd-order Taylor approximation of exp(q.k/sqrt(dk))
            def phi(t):
                t = t / self.width ** 0.25
                return torch.cat([torch.ones_like(t[..., :1]), t,
                                  (t.unsqueeze(-1) * t.unsqueeze(-2)).flatten(-2) / math.sqrt(2)], -1)
            y = LINEAR_IMPL(phi(q), phi(k), v)
        elif self.kind in ("cpla", "cp-softmax"):
            pq, pk = math.pi * torch.tanh(q), math.pi * torch.tanh(k)
            # CP kernel computed pairwise (identical to the 3^n / (2p+1)^n feature-map inner product)
            s = torch.cos((pq.unsqueeze(-2) - pk.unsqueeze(-3)) / 2).pow(2 * self.power).prod(-1)
            if self.kind == "cpla":
                s = s * causal
                y = (s @ v) / (s.sum(-1, keepdim=True) + 1e-6)
            else:
                y = F.softmax((s * self.log_scale.exp()).masked_fill(~causal, -1e9), -1) @ v
        elif self.kind == "elu":
            y = LINEAR_IMPL(F.elu(q) + 1, F.elu(k) + 1, v)
        elif self.kind == "performer":
            def phi(t, stabilize):
                t = t / self.width ** 0.25
                proj = torch.einsum("bhnd,hdm->bhnm", t, self.proj)
                z = proj - t.pow(2).sum(-1, keepdim=True) / 2
                if stabilize:   # per-query shift cancels exactly in the normalized output (no leakage)
                    z = z - z.amax(-1, keepdim=True).detach()
                return torch.exp(z) / math.sqrt(self.m)
            y = LINEAR_IMPL(phi(q, True), phi(k, False), v)
        else:
            raise ValueError(self.kind)
        return self.o(y.transpose(1, 2).reshape(B, N, -1))


class Block(nn.Module):
    def __init__(self, d, heads, kind):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads, kind)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class RecallModel(nn.Module):
    def __init__(self, vocab, d, heads, layers, kind, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(0.02 * torch.randn(max_len, d))
        kinds = kind.split("+") if "+" in kind else [kind] * layers   # e.g. "softmax+cpla"
        self.blocks = nn.ModuleList(Block(d, heads, k) for k in kinds)
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.emb(ids) + self.pos[: ids.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_batch(B, K, Q, n_keys, n_vals, g):
    """Keys use ids [0, n_keys); values use ids [n_keys, n_keys + n_vals)."""
    keys = torch.argsort(torch.rand(B, n_keys, generator=g), 1)[:, :K]
    vals = n_keys + torch.randint(0, n_vals, (B, K), generator=g)
    kv = torch.stack([keys, vals], -1).reshape(B, 2 * K)
    pick = torch.argsort(torch.rand(B, K, generator=g), 1)[:, :Q]
    queries = torch.gather(keys, 1, pick)
    answers = torch.gather(vals, 1, pick)
    ids = torch.cat([kv, queries], 1)
    targets = torch.full_like(ids, -100)
    targets[:, 2 * K:] = answers
    return ids, targets


def run(kind, K, seed, d=64, heads=4, layers=2, steps=2000, B=64, n_keys=64, n_vals=64,
        lr=1e-3, Q=None):
    Q = Q or min(K, 16)
    torch.manual_seed(seed)
    model = RecallModel(n_keys + n_vals, d, heads, layers, kind, 2 * K + Q)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(1000 + seed)
    t0 = time.time()
    for _ in range(steps):
        ids, tgt = make_batch(B, K, Q, n_keys, n_vals, g)
        loss = F.cross_entropy(model(ids).flatten(0, 1), tgt.flatten(), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    gt = torch.Generator().manual_seed(99_000 + seed)                # held-out sequences
    correct = total = 0
    with torch.no_grad():
        for _ in range(8):
            ids, tgt = make_batch(256, K, Q, n_keys, n_vals, gt)
            pred = model(ids).argmax(-1)
            m = tgt != -100
            correct += (pred[m] == tgt[m]).sum().item(); total += m.sum().item()
    return dict(model=kind, K=K, seed=seed, test_acc=correct / total, final_loss=loss.item(),
                params=sum(p.numel() for p in model.parameters()),
                seconds=round(time.time() - t0, 1))


def write_csv(name, rows):
    path = os.path.join(OUT, name)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["softmax", "cp-softmax", "cpla", "elu", "performer"]
    Ks = [int(k) for k in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8, 16, 32]
    seeds = range(int(sys.argv[3])) if len(sys.argv) > 3 else range(3)
    out = sys.argv[4] if len(sys.argv) > 4 else "recall.csv"
    for K in Ks:
        for s in seeds:
            for kind in kinds:
                r = run(kind, K, s)
                print(r, flush=True)
                write_csv(out, [r])
