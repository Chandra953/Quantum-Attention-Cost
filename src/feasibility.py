"""
Feasibility map: quantum hardware vs exact classical evaluation of fidelity attention.

For a register of n qubits, compare the cost of ONE training step on ONE sequence of
length N:

Quantum hardware (pairwise overlaps, parameter-shift gradients):
    circuits = N^2 * (1 + 2P)          P = 2n + 2nL trainable+encoded angles per score
    shots    = circuits * S(n),        S(n) = 1 / Var[grad](n)  (resolve a typical gradient at SNR 1)
    time     = shots / shot_rate

Exact classical evaluation (Proposition 2: explicit 4^n features, linear in N):
    flops  = N * (2 * g(n) * 2^n + 2 * 4^n * d) * 3     g(n) = n(L+1) gates per state; x3 for backward
    memory = 4^n * d * 8 bytes  (prefix-sum state per head)
    time   = flops / flop_rate

Var[grad](n) is the exponential fit to the measured gradient variance of the
quantum layer (Paper 1 data, L = 2): Var ~ a * r^n. Extrapolating the fit
beyond n = 10 is an explicit assumption of this analysis.
"""
import csv
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GRAD = os.path.join(HERE, "..", "..", "quantum-fidelity-attention", "results", "gradient_variance.csv")


def fit_variance(path=GRAD, L=2, n_min=4):
    g = pd.read_csv(path)
    s = g[(g.L == L) & (g.n >= n_min)]
    b, a = np.polyfit(s.n, np.log(s.grad_var), 1)
    return math.exp(a), math.exp(b)


def costs(n, N=512, d=64, L=2, shot_rate=1e4, flop_rate=1e13, var=None):
    a, r = var
    P = 2 * n + 2 * n * L
    shots = N ** 2 * (1 + 2 * P) / (a * r ** n)
    q_time = shots / shot_rate
    flops = N * (2 * n * (L + 1) * 2 ** n + 2 * 4 ** n * d) * 3
    c_time = flops / flop_rate
    c_mem = 4 ** n * d * 8
    return dict(n=n, quantum_shots=shots, quantum_seconds=q_time,
                classical_flops=flops, classical_seconds=c_time, classical_bytes=c_mem)


if __name__ == "__main__":
    var = fit_variance()
    print(f"gradient variance fit: Var ~ {var[0]:.3g} * {var[1]:.4f}^n  ({1 / var[1]:.2f}x smaller per qubit)")
    rows = [costs(n, var=var) for n in range(2, 41, 2)]
    out = os.path.join(HERE, "..", "results", "feasibility.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    day = 86400
    for r in rows:
        print(f"n={r['n']:2d}  quantum {r['quantum_seconds']:9.2e} s   classical {r['classical_seconds']:9.2e} s"
              f"  classical memory {r['classical_bytes'] / 1e9:9.2e} GB")
    # Is there any n where quantum fits a 1-day budget but classical does not (time or 1 TB memory)?
    window = [r["n"] for r in rows if r["quantum_seconds"] < day and
              (r["classical_seconds"] > day or r["classical_bytes"] > 1e12)]
    print("n where quantum is feasible (<1 day/step) but exact classical is not:", window or "none")
