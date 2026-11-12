"""Checks that the code implements what Paper 2 states, and that the reported numbers
follow from the committed data. Run from the repository root:  pytest -q
"""
import csv
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)                      # recall.py writes to ./results

import feasibility as FE  # noqa: E402
import recall as R  # noqa: E402

torch.set_default_dtype(torch.float64)


def cp_kernel(a, b):
    """s(a, b) = prod_k cos^2((a_k - b_k) / 2), the cosine-product kernel."""
    return torch.cos((a - b) / 2).pow(2).prod(-1)


# ---------------------------------------------------------------------------
# Proposition 1: exact 3^n feature map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_feature_map_dimension_is_3_to_the_n(n):
    assert R.cp_features(torch.zeros(n)).shape[-1] == 3 ** n


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_feature_map_reproduces_cp_kernel_exactly(n):
    g = torch.Generator().manual_seed(n)
    a = (torch.rand(50, n, generator=g) * 2 - 1) * math.pi
    b = (torch.rand(50, n, generator=g) * 2 - 1) * math.pi
    lhs = (R.cp_features(a) * R.cp_features(b)).sum(-1)
    assert torch.allclose(lhs, cp_kernel(a, b), atol=1e-12)


def test_cp_kernel_is_bounded_and_one_on_the_diagonal():
    a = torch.rand(20, 4) * 2 * math.pi
    b = torch.rand(20, 4) * 2 * math.pi
    s = cp_kernel(a, b)
    assert torch.all((s >= 0) & (s <= 1))
    assert torch.allclose(cp_kernel(a, a), torch.ones(20))


# ---------------------------------------------------------------------------
# Linear-time attention equals its pairwise form
# ---------------------------------------------------------------------------

def test_prefix_sum_attention_equals_pairwise():
    g = torch.Generator().manual_seed(0)
    phi_q = torch.rand(2, 3, 17, 9, generator=g)
    phi_k = torch.rand(2, 3, 17, 9, generator=g)
    v = torch.randn(2, 3, 17, 5, generator=g)
    assert torch.allclose(R.causal_linear(phi_q, phi_k, v), R.causal_pairwise(phi_q, phi_k, v), atol=1e-10)


def test_cpla_layer_matches_feature_map_attention():
    """The CPLA layer computes the kernel pairwise; it must equal linear attention on cp_features."""
    torch.manual_seed(0)
    att = R.Attention(d=16, heads=2, kind="cpla", n_angles=3)
    x = torch.randn(1, 11, 16)
    B, N, _ = x.shape
    q, k = att.split(att.q(x), att.width), att.split(att.k(x), att.width)
    v = att.split(att.v(x), att.dv)
    fq, fk = R.cp_features(math.pi * torch.tanh(q)), R.cp_features(math.pi * torch.tanh(k))
    y = R.causal_linear(fq, fk, v)
    expected = att.o(y.transpose(1, 2).reshape(B, N, -1))
    assert torch.allclose(att(x), expected, atol=1e-8)


# ---------------------------------------------------------------------------
# No attention variant may look at future tokens
# ---------------------------------------------------------------------------

KINDS = ["softmax", "window-w4", "cpla", "cp-softmax", "taylor-d8", "elu-d81", "performer-d16"]


@pytest.mark.parametrize("kind", KINDS)
def test_attention_is_causal(kind):
    torch.manual_seed(1)
    att = R.Attention(d=16, heads=2, kind=kind)
    x = torch.randn(1, 12, 16)
    x2 = x.clone()
    x2[:, 7:] = torch.randn(1, 5, 16)          # change only the future
    assert torch.allclose(att(x)[:, :7], att(x2)[:, :7], atol=1e-9), kind


# ---------------------------------------------------------------------------
# Associative-recall data
# ---------------------------------------------------------------------------

def test_recall_batches_have_correct_answers():
    g = torch.Generator().manual_seed(3)
    K, Q, n_keys = 8, 5, 64
    ids, tgt = R.make_batch(4, K, Q, n_keys, 64, g)
    assert ids.shape == (4, 2 * K + Q)
    assert torch.all(tgt[:, : 2 * K] == -100)
    for b in range(4):
        pairs = {ids[b, 2 * i].item(): ids[b, 2 * i + 1].item() for i in range(K)}
        assert len(pairs) == K                                   # keys are distinct
        for j in range(Q):
            assert tgt[b, 2 * K + j].item() == pairs[ids[b, 2 * K + j].item()]


# ---------------------------------------------------------------------------
# Cost model (Theorem 1 and Section V)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,L", [(4, 2), (10, 4), (30, 4)])
def test_circuits_per_pair(n, L):
    P = 2 * n * (L + 1)
    assert FE.circuits_per_pair(n, L, "shift") == 1 + 2 * P
    assert FE.circuits_per_pair(n, L, "ideal") == 3


def test_quantum_shots_respect_the_theorem_lower_bound():
    f = FE.fits(4)
    for n in (10, 20, 30):
        for N in (64, 4096):
            kappa = FE.circuits_per_pair(n, 4)
            q = FE.quantum_shots(n, N, 4, f)
            assert q >= kappa * N ** 2
            assert FE.quantum_shots(n, N, 4, f, coherent=False) >= q


def test_classical_cost_is_infinite_when_nothing_fits_in_memory():
    assert FE.classical(60, 1024, 4, 64, 1e15, 1e12) == math.inf
    assert math.isfinite(FE.classical(10, 1024, 4, 64, 1e15, 1e12))


# ---------------------------------------------------------------------------
# Reported numbers follow from the committed data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L,grad_rate,score_rate", [(2, 3.53, 1.99), (4, 3.97, 2.01)])
def test_fitted_decay_rates_match_paper(L, grad_rate, score_rate):
    f = FE.fits(L)
    assert round(1 / f["rV"], 2) == grad_rate
    assert round(1 / f["rp"], 2) == score_rate


def test_sensitivity_sweep_is_reproducible():
    path = os.path.join(ROOT, "results", "feasibility_sensitivity.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 144
    for row in rows[::12]:                     # spot-check every 12th scenario
        _, best = FE.scan(L=int(row["L"]), shot_rate=float(row["shot_rate"]),
                          flop_rate=float(row["flop_rate"]), mem=float(row["memory_bytes"]),
                          coherent=row["coherent"] == "True", gradient=row["gradient"])
        expected = float(row["min_quantum_seconds"])
        if best is None:
            assert math.isinf(expected)
        else:
            assert best[0] == pytest.approx(expected, rel=1e-9)
            assert best[1] == int(row["at_n"]) and best[2] == int(row["at_N"])


def test_headline_recall_numbers():
    rows = []
    for name in os.listdir(os.path.join(ROOT, "results")):
        if name.startswith("recall_") and name.endswith(".csv"):
            with open(os.path.join(ROOT, "results", name)) as fh:
                rows += list(csv.DictReader(fh))

    def mean_acc(layer2, K):
        acc = [float(r["test_acc"]) for r in rows
               if r["model"].split("+")[-1] == layer2 and int(r["K"]) == K]
        assert acc, (layer2, K)
        return np.mean(acc)

    assert round(mean_acc("cpla", 16), 2) == 0.86
    assert round(mean_acc("elu-d81", 16), 3) == 0.125


def test_cpla_feature_map_path_equals_pairwise_path():
    """The fast feature-map CPLA path (used for language modelling) computes the same function."""
    torch.manual_seed(2)
    att = R.Attention(d=16, heads=2, kind="cpla")
    x = torch.randn(2, 13, 16)
    y_pair = att(x)
    att.use_features = True
    assert torch.allclose(att(x), y_pair, atol=1e-9)


def test_hardware_results_match_their_circuits():
    """The exact scores stored with the IBM hardware run are regenerated from the script's seed,
    and the headline agreement statistics follow from the stored data."""
    import pandas as pd
    import hardware_ibm as H
    d = pd.read_csv(os.path.join(ROOT, "results", "hardware_ibm.csv"))
    jobs = H.build(sorted(d.n.unique()), int(d.L.iloc[0]), 20, 2026)
    assert np.allclose([j["exact"] for j in jobs], d.exact.values, atol=1e-12)
    assert round(np.corrcoef(d.exact, d.measured)[0, 1], 3) == 0.998
    assert (d.measured - d.exact).abs().mean() == pytest.approx(0.007, abs=5e-4)


@pytest.mark.parametrize("L,grad_rate,score_rate", [(2, 3.39, 2.01), (4, 3.70, 2.01)])
def test_design_b_rates_match_paper(L, grad_rate, score_rate):
    f = FE.fits(L, os.path.join(ROOT, "results", "scaling_design_b.csv"))
    assert round(1 / f["rV"], 2) == grad_rate
    assert round(1 / f["rp"], 2) == score_rate


def test_design_b_sweep_within_factor_3_4_of_design_a():
    import pandas as pd
    k = ["gradient", "coherent", "shot_rate"]
    a = pd.read_csv(os.path.join(ROOT, "results", "feasibility_sensitivity.csv")).groupby(k).min_quantum_seconds.min()
    b = pd.read_csv(os.path.join(ROOT, "results", "feasibility_sensitivity_design_b.csv")).groupby(k).min_quantum_seconds.min()
    r = b / a
    assert len(r) == 12
    assert round(max(r.max(), 1 / r.min()), 1) == 3.4


def test_wikitext_table_matches_results():
    import pandas as pd
    d = pd.read_csv(os.path.join(ROOT, "results", "lm_wikitext2.csv"))
    g = d.groupby("model")
    assert (g.size() == 2).all() and len(g) == 6
    m, lr = g.test_bpc.mean(), g.test_longrange_bpc.mean()
    table = {"window-w64": (1.899, 1.236), "softmax": (1.906, 1.241), "performer-d16": (1.981, 1.343),
             "elu-d81": (1.986, 1.355), "taylor-d8": (2.016, 1.379), "cpla": (2.081, 1.455)}
    for k, (a, b) in table.items():
        assert round(m[k], 3) == a and round(lr[k], 3) == b, k
    assert round(m["cpla"] - m["elu-d81"], 3) == 0.095 and round(lr["cpla"] - lr["elu-d81"], 3) == 0.100
    assert round(m["cpla"] - m["taylor-d8"], 3) == 0.065 and round(lr["cpla"] - lr["taylor-d8"], 3) == 0.076
    assert (d.test_longrange_positions == 16753).all()


def test_compiled_circuit_sizes_match_paper():
    """Table: depth and CZ count of the hardware circuits compiled for ibm_fez (reference compilation)."""
    fake = pytest.importorskip("qiskit_ibm_runtime.fake_provider")
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    import hardware_ibm as H
    pm = generate_preset_pass_manager(optimization_level=3, backend=fake.FakeFez(), seed_transpiler=0)
    jobs = H.build([2, 3, 4, 5, 6, 8], 2, 20, 2026)
    expected = {2: (13, 2), 3: (43, 12), 4: (46, 16), 5: (59, 22), 6: (123, 48), 8: (115, 40)}
    for n, (depth, cz) in expected.items():
        circs = [pm.run(j["circuit"]) for j in jobs if j["n"] == n]
        assert round(np.mean([c.depth() for c in circs])) == depth, n
        assert round(np.mean([c.count_ops().get("cz", 0) for c in circs])) == cz, n
