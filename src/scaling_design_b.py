"""
Scaling measurements for a second circuit design (robustness check for Section IV).

Design A (Paper 1 / scaling.py): RY angle encoding, then L layers of
    RY on every qubit + CNOT ring.            Real amplitudes, n*L parameters per state.
Design B (this file): RX angle encoding, then L layers of
    RY and RZ on every qubit + CZ chain.      Complex amplitudes, 2*n*L parameters per state.

The score is the same fidelity s = |<psi(a, theta_Q) | psi(b, theta_K)>|^2.
For n = 2..12 and L in {2, 4}, at uniformly random inputs and parameters, we record
Var[ds/dtheta_Q,1] and E[s], then fit Var ~ a r^n with bootstrap confidence intervals
(the same protocol as scaling.py). Writes results/scaling_design_b.csv.
"""
import csv
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scaling import fit  # noqa: E402

CDT = torch.complex128


def _apply_1q(st, U, k, n):
    """Apply a batch of 2x2 gates U (M, 2, 2) to qubit k of states st (M, 2, ..., 2)."""
    st = st.movedim(k + 1, -1)
    st = torch.einsum("m...j,mij->m...i", st, U)
    return st.movedim(-1, k + 1)


def _rx(t):
    c, s = torch.cos(t / 2).to(CDT), torch.sin(t / 2).to(CDT)
    return torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)


def _ry(t):
    c, s = torch.cos(t / 2).to(CDT), torch.sin(t / 2).to(CDT)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def _rz(t):
    e = torch.exp(-0.5j * t.to(CDT))
    z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


def _cz_phase(n):
    """Diagonal of the CZ chain (0-1, 1-2, ..., (n-2)-(n-1)) as a (2,)*n tensor of +-1."""
    idx = torch.arange(2 ** n)
    bits = [(idx >> (n - 1 - k)) & 1 for k in range(n)]
    parity = sum(bits[k] * bits[k + 1] for k in range(n - 1)) if n > 1 else torch.zeros_like(idx)
    return (1 - 2 * (parity % 2)).to(CDT).reshape((2,) * n)


def simulate_b(angles, theta, n, L):
    """|psi> = prod_l [CZ chain . RZ . RY] . RX(angles) |0>.  angles: (M, n), theta: (2nL,)."""
    M = angles.shape[0]
    st = torch.zeros((M,) + (2,) * n, dtype=CDT)
    st.view(M, -1)[:, 0] = 1.0
    for k in range(n):
        st = _apply_1q(st, _rx(angles[:, k]), k, n)
    cz = _cz_phase(n)
    for l in range(L):
        for k in range(n):
            st = _apply_1q(st, _ry(theta[2 * (l * n + k)].expand(M)), k, n)
            st = _apply_1q(st, _rz(theta[2 * (l * n + k) + 1].expand(M)), k, n)
        st = st * cz
    return st.reshape(M, -1)


def measure(n, L, samples, seed=0):
    g = torch.Generator().manual_seed(seed)
    grads, scores = [], []
    for _ in range(samples):
        a = torch.rand(1, n, generator=g, dtype=torch.float64) * 2 * math.pi
        b = torch.rand(1, n, generator=g, dtype=torch.float64) * 2 * math.pi
        tq = (torch.rand(2 * n * L, generator=g, dtype=torch.float64) * 2 * math.pi).requires_grad_()
        tk = torch.rand(2 * n * L, generator=g, dtype=torch.float64) * 2 * math.pi
        amp = (simulate_b(a, tq, n, L).conj() * simulate_b(b, tk, n, L)).sum()
        s = amp.abs() ** 2
        (gq,) = torch.autograd.grad(s, tq)
        grads.append(gq[0].item()); scores.append(s.item())
    return np.array(grads), np.array(scores)


if __name__ == "__main__":
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = os.path.join(HERE, "..", "results", "scaling_design_b.csv")
    rows = []
    for L in (2, 4):
        for n in range(2, 13):
            gr, sc = measure(n, L, samples, seed=7000 + 100 * L + n)
            rows.append(dict(n=n, L=L, samples=samples, grad_var=float(gr.var()),
                             score_mean=float(sc.mean()), score_var=float(sc.var())))
            print(rows[-1], flush=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for L in (2, 4):
        sub = [r for r in rows if r["L"] == L and r["n"] >= 4]
        ns = [r["n"] for r in sub]
        for key in ("grad_var", "score_mean"):
            a, r, lo, hi = fit(ns, [r_[key] for r_ in sub])
            print(f"L={L} {key}: 1/r = {1/r:.2f}, 95% CI {1/hi:.2f}-{1/lo:.2f}")
