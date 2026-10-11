"""Figures for Paper 2, generated from ../results/*.csv. Writes PDFs (paper) and PNGs (README)."""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "figures")
ASSETS = os.path.join(HERE, "..", "assets")
os.makedirs(FIG, exist_ok=True); os.makedirs(ASSETS, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import feasibility as FE  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
BLUE, ORANGE, GREEN, GREY, RED = "#1f5fbf", "#d4622a", "#2e8b57", "#9a9a9a", "#b33a3a"
COL, DCOL = 3.5, 7.16


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(ASSETS, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_scaling():
    s = pd.read_csv(os.path.join(RES, "scaling.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75))
    for L, alpha in [(2, 0.5), (4, 1.0)]:
        d = s[s.L == L]
        axes[0].semilogy(d.n, d.grad_var, "o-", color=BLUE, alpha=alpha, ms=2.5, lw=1, label=f"$L={L}$")
        axes[1].semilogy(d.n, d.score_mean, "o-", color=ORANGE, alpha=alpha, ms=2.5, lw=1, label=f"$L={L}$")
    n = np.arange(2, 13)
    axes[0].semilogy(n, 1.0 * 4.0 ** -n, "--", color=GREY, lw=0.8, label=r"$4^{-n}$")
    axes[1].semilogy(n, 2.0 ** -n, "--", color=GREY, lw=0.8, label=r"$2^{-n}$")
    axes[0].set_ylabel(r"Var$[\partial s/\partial\theta]$"); axes[1].set_ylabel(r"$\mathbb{E}[s]$")
    for ax in axes:
        ax.set_xlabel("Qubits $n$"); ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.legend(frameon=False, fontsize=6); ax.grid(color="#e5e5e5", lw=0.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "scaling")


def fig_feasibility():
    L, d, R_s, F, mem = 4, 64, 1e6, 1e13, 1e15
    f = FE.fits(L)
    ns = np.arange(10, 51)
    logNs = np.arange(6, 25)
    ratio = np.zeros((len(logNs), len(ns)))
    for i, lN in enumerate(logNs):
        for j, n in enumerate(ns):
            tq = FE.quantum_shots(int(n), 2 ** int(lN), L, f) / R_s
            tc = FE.classical(int(n), 2 ** int(lN), L, d, F, mem)
            ratio[i, j] = math.log10(tq / tc) if math.isfinite(tc) else np.nan
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    im = ax.pcolormesh(ns, logNs, ratio, cmap="RdBu_r", vmin=-6, vmax=6, shading="nearest")
    ax.contour(ns, logNs, ratio, levels=[0], colors="k", linewidths=0.9)
    nan_cols = ns[np.isnan(ratio).all(axis=0)]
    if len(nan_cols):
        ax.axvspan(nan_cols.min() - 0.5, ns.max() + 0.5, facecolor="none", hatch="////",
                   edgecolor="#bbbbbb", lw=0)
        ax.text((nan_cols.min() + ns.max()) / 2 + 0.5, 15, "classical > 1 PB", fontsize=5.5,
                ha="center", va="center", rotation=90, color="#555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    _, best = FE.scan(L=L, d=d, shot_rate=R_s, flop_rate=F, mem=mem)
    ax.plot(best[1], math.log2(best[2]), "k*", ms=7)
    ax.annotate(f"fastest advantage:\n{best[0] / 86400:.1f} days per step", (best[1], math.log2(best[2])),
                (14.5, 9.0), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.6),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("Qubits $n$"); ax.set_ylabel(r"Sequence length $\log_2 N$")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}(T_{\mathrm{quantum}}/T_{\mathrm{classical}})$", fontsize=7)
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    ax.text(11, 22.6, "classical cheaper", fontsize=6.5, color="#7a1f1f", bbox=box)
    ax.text(33, 21.5, "quantum cheaper", fontsize=6.5, color="#1f3f7a", bbox=box)
    ax.text(30.5, 11.2, "black line: equal cost", fontsize=6, color="k", bbox=box)
    fig.tight_layout()
    save(fig, "feasibility")


def fig_shotrate():
    s = pd.read_csv(os.path.join(RES, "feasibility_sensitivity.csv"))
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    for grad, coh, c, ls, lab in [("shift", True, BLUE, "o-", "parameter shift, coherent"),
                                  ("shift", False, BLUE, "o:", "parameter shift, incoherent"),
                                  ("ideal", True, ORANGE, "s-", "ideal gradients, coherent"),
                                  ("ideal", False, ORANGE, "s:", "ideal gradients, incoherent")]:
        g = s[(s.gradient == grad) & (s.coherent == coh)].groupby("shot_rate").min_quantum_seconds.min() / 86400
        ax.loglog(g.index, g.values, ls, color=c, ms=3, lw=1.1, label=lab)
    for y, t in [(1 / 24, "1 hour"), (1, "1 day"), (365, "1 year")]:
        ax.axhline(y, color="#cccccc", lw=0.6); ax.text(1.12e6, y, t, fontsize=5.5, color="#888", va="center")
    ax.set_xlabel("Hardware shot rate (shots/s)")
    ax.set_ylabel("Min. days per training step")
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              columnspacing=0.8)
    ax.set_xlim(8e3, 1.25e6)
    ax.grid(color="#e5e5e5", lw=0.5, which="major")
    fig.tight_layout()
    save(fig, "shotrate")


def fig_recall():
    d = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in os.listdir(RES)
                   if f.startswith("recall_") and f.endswith(".csv")])
    d["layer2"] = d.model.str.split("+").str[1]
    d = d[d.K <= 32]                      # K = 64 exceeded the training budget even for softmax
    style = {"softmax": ("Softmax (quadratic)", "k", "s--"),
             "cpla": ("CPLA (81)", GREEN, "o-"),
             "cp-softmax": ("CP kernel, softmax", ORANGE, "o:"),
             "taylor-d16": ("Based, $d'=16$ (273)", BLUE, "^-"),
             "taylor-d8": ("Based, $d'=8$ (73)", BLUE, "^:"),
             "elu-d81": ("Linear elu+1 (81)", RED, "v-"),
             "performer-d16": ("Performer", GREY, "x:"),
             "window-w4": ("No global layer", GREY, "-")}
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    for k, (lab, c, ls) in style.items():
        g = d[d.layer2 == k].groupby("K").test_acc
        if g.ngroups == 0:
            continue
        m, sd = g.mean(), g.std().fillna(0)
        ax.errorbar(m.index, m.values, yerr=sd.values, fmt=ls, color=c, ms=3, lw=1, capsize=1.5, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Key-value pairs $K$"); ax.set_ylabel("Recall accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=5.8, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              columnspacing=0.8, handlelength=2.2)
    ax.grid(color="#e5e5e5", lw=0.5)
    fig.tight_layout()
    save(fig, "recall")


if __name__ == "__main__":
    which = sys.argv[1:] or ["scaling", "feasibility", "shotrate", "recall"]
    for w in which:
        globals()["fig_" + w]()
    print("done")
