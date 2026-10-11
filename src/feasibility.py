"""
Cost of training fidelity attention: quantum hardware vs exact classical simulation.

Setting: one training step on one sequence of N tokens, n qubits, L ansatz layers,
P = 2n(L+1) angles per score circuit (2n encoded + 2nL trainable), value width d.

QUANTUM (Theorem 1, most favourable to quantum):
  Every score needs >= 1 shot for each of its 1 + 2P circuits (forward + parameter shift).
  A shifted score s is a Bernoulli mean with variance p(1-p)/S, p ~ E[s].
  Even if per-score gradients add perfectly coherently across the N^2 pairs, resolving the
  gradient of a shared parameter at SNR >= 1 needs S >= p / (2 N^2 V) shots per circuit,
  where V = Var[ds/dtheta]. Hence
      shots >= (1 + 2P) * max(N^2, p / (2 V)).
  (Incoherent aggregation multiplies the second term by N^2.)

CLASSICAL (exact, Propositions 1-2 of Paper 1), cheaper of two algorithms:
  pairwise : prepare N states (2^n amplitudes), N^2 overlaps      time ~ 6 [2 N g 2^n + 2 N^2 2^n], memory ~ 2 * 2^n * 16 B
  linear   : explicit 4^n features, prefix sums                   time ~ 6 [2 N g 2^n + 4 N 4^n d],  memory ~ 4^n d * 8 B
  g = n (L + 1) gates per state; factor 6 ~ forward + backward. Streaming keeps memory per head.

V(n) and p(n) are exponential fits to measured data (results/scaling.csv); extrapolation
beyond n = 12 is an explicit assumption.
"""
import csv
import itertools
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCALING = os.path.join(HERE, "..", "results", "scaling.csv")
DAY, YEAR = 86400.0, 3.156e7


def fits(L):
    s = pd.read_csv(SCALING)
    s = s[(s.L == L) & (s.n >= 4)]
    bv, av = np.polyfit(s.n, np.log(s.grad_var), 1)
    bp, ap = np.polyfit(s.n, np.log(s.score_mean), 1)
    return dict(aV=math.exp(av), rV=math.exp(bv), ap=math.exp(ap), rp=math.exp(bp))


def circuits_per_pair(n, L, gradient="shift"):
    """kappa: circuits per token pair per training step.
    'shift': parameter-shift rule, 1 + 2P with P = 2n(L+1).
    'ideal': a hypothetical backpropagation-like estimator that returns all gradients
             from 2 extra circuits (cf. Bowles et al., Quantum 9, 1873, 2025)."""
    return 1 + 2 * 2 * n * (L + 1) if gradient == "shift" else 3


def quantum_shots(n, N, L, f, coherent=True, gradient="shift"):
    kappa = circuits_per_pair(n, L, gradient)
    V = f["aV"] * f["rV"] ** n
    p = f["ap"] * f["rp"] ** n
    need = p / (2 * V)
    return kappa * (max(N ** 2, need) if coherent else N ** 2 * max(1.0, need))


def classical(n, N, L, d, flop_rate, mem):
    g = n * (L + 1)
    options = []
    pair_t = 6 * (2 * N * g * 2 ** n + 2 * N ** 2 * 2 ** n)
    if 2 * 2 ** n * 16 <= mem:
        options.append(pair_t / flop_rate)
    lin_t = 6 * (2 * N * g * 2 ** n + 4 * N * 4 ** n * d)
    if 4 ** n * d * 8 <= mem:
        options.append(lin_t / flop_rate)
    return min(options) if options else math.inf


def scan(L=4, d=64, shot_rate=1e4, flop_rate=1e13, mem=1e12, coherent=True, gradient="shift",
         ns=range(2, 61), Ns=tuple(2 ** k for k in range(6, 25))):
    f = fits(L)
    Ns = list(Ns)
    best = None
    for n, N in itertools.product(ns, Ns):
        tq = quantum_shots(n, N, L, f, coherent, gradient) / shot_rate
        tc = classical(n, N, L, d, flop_rate, mem)
        if tq < tc and (best is None or tq < best[0]):
            best = (tq, n, N, tc)
    return f, best


if __name__ == "__main__":
    rows = []
    grid = dict(L=[2, 4], shot_rate=[1e4, 1e5, 1e6], flop_rate=[1e13, 1e15, 1e17],
                mem=[1e12, 1e15], coherent=[True, False], gradient=["shift", "ideal"])
    for L, sr, fr, mem, coh, grad in itertools.product(*grid.values()):
        f, best = scan(L=L, shot_rate=sr, flop_rate=fr, mem=mem, coherent=coh, gradient=grad)
        row = dict(L=L, shot_rate=sr, flop_rate=fr, memory_bytes=mem, coherent=coh, gradient=grad,
                   decay_per_qubit=round(1 / f["rV"], 3))
        if best:
            row.update(min_quantum_seconds=best[0], min_quantum_years=best[0] / YEAR,
                       at_n=best[1], at_N=best[2], classical_seconds_there=best[3])
        else:
            row.update(min_quantum_seconds=math.inf, min_quantum_years=math.inf,
                       at_n=None, at_N=None, classical_seconds_there=None)
        rows.append(row)
    out = os.path.join(HERE, "..", "results", "feasibility_sensitivity.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    summary = df.groupby(["gradient", "coherent", "shot_rate"]).min_quantum_seconds.min()
    print("Fastest quantum-advantage training step (seconds), min over L, flop rate, memory:")
    print(summary.to_string())
