"""
Scaling measurements for the quantum fidelity attention score (Paper 1 circuit).

For n = 2..12 qubits and L in {2, 4} ansatz layers, at uniformly random parameters:
  * gradient variance   Var[ d s / d theta_Q,1 ]
  * kernel concentration Var[ s ] and E[ s ] over random input pairs
using the exact statevector simulator from Paper 1 (validated against Qiskit).
Writes results/scaling.csv and fits Var ~ a * r^n with bootstrap confidence intervals.
"""
import csv
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "quantum-fidelity-attention", "src"))
from experiments import simulate  # noqa: E402

torch.set_default_dtype(torch.float64)


def measure(n, L, samples, seed=0):
    g = torch.Generator().manual_seed(seed)
    grads, scores = [], []
    for _ in range(samples):
        a = torch.rand(1, n, generator=g) * 2 * math.pi
        b = torch.rand(1, n, generator=g) * 2 * math.pi
        tq = (torch.rand(n * L, generator=g) * 2 * math.pi).requires_grad_()
        tk = torch.rand(n * L, generator=g) * 2 * math.pi
        s = ((simulate(a, tq, n, L) * simulate(b, tk, n, L)).sum()) ** 2
        (gq,) = torch.autograd.grad(s, tq)
        grads.append(gq[0].item()); scores.append(s.item())
    return np.array(grads), np.array(scores)


def fit(ns, values, boot=2000, seed=0):
    """Fit log(values) = log a + n log r; bootstrap over the per-n points for a 95% CI on r."""
    b, a = np.polyfit(ns, np.log(values), 1)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(boot):
        idx = rng.integers(0, len(ns), len(ns))
        if len(set(np.asarray(ns)[idx])) < 2:
            continue
        rs.append(np.polyfit(np.asarray(ns)[idx], np.log(np.asarray(values)[idx]), 1)[0])
    lo, hi = np.percentile(np.exp(rs), [2.5, 97.5])
    return math.exp(a), math.exp(b), lo, hi


if __name__ == "__main__":
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rows = []
    for L in (2, 4):
        for n in range(2, 13):
            gr, sc = measure(n, L, samples, seed=100 * L + n)
            rows.append(dict(n=n, L=L, samples=samples, grad_var=float(gr.var()),
                             score_mean=float(sc.mean()), score_var=float(sc.var())))
            print(rows[-1], flush=True)
    out = os.path.join(HERE, "..", "results", "scaling.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for L in (2, 4):
        sub = [r for r in rows if r["L"] == L and r["n"] >= 4]
        ns = [r["n"] for r in sub]
        for key in ("grad_var", "score_var"):
            a, r, lo, hi = fit(ns, [r_[key] for r_ in sub])
            print(f"L={L} {key}: {a:.3g} * {r:.4f}^n  (1/r = {1/r:.2f}, 95% CI for 1/r: {1/hi:.2f}-{1/lo:.2f})")
