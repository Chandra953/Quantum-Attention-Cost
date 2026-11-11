"""
Measure the fidelity attention score on IBM quantum hardware (Section IV check).

For n qubits we build the compute-uncompute circuit of Paper 1:
    |0> --U(a, theta_Q)-- U(b, theta_K)^dagger -- measure,   P(0...0) = |<psi(b)|psi(a)>|^2 = s,
with U = RY angle encoding followed by L layers of (RY on every qubit + CNOT ring).
For random inputs and parameters we compare the hardware estimate of s with the exact value
and check that the mean score decays as about 2^-n.

Usage
  pip install qiskit qiskit-ibm-runtime pandas
  python src/hardware_ibm.py --dry-run                 # noiseless local check, no account needed
  export QISKIT_IBM_TOKEN=<your API key>               # never commit or share the key
  python src/hardware_ibm.py --backend least_busy      # real hardware, ~2-3 min of QPU time

Writes results/hardware_ibm.csv (one row per circuit) and prints the mean score per n.
"""
import argparse
import csv
import math
import os

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(HERE, "..", "results")
# inside the repository: results/hardware_ibm.csv; script downloaded on its own: next to the script
OUT = os.path.join(_RESULTS if os.path.isdir(_RESULTS) else HERE, "hardware_ibm.csv")


def state_circuit(x, theta, n, L):
    qc = QuantumCircuit(n)
    for k in range(n):
        qc.ry(float(x[k]), k)
    for l in range(L):
        for k in range(n):
            qc.ry(float(theta[l * n + k]), k)
        if n > 1:
            for k in range(n):
                qc.cx(k, (k + 1) % n)
    return qc


def fidelity_circuit(a, b, tq, tk, n, L):
    qc = state_circuit(a, tq, n, L)
    qc.compose(state_circuit(b, tk, n, L).inverse(), inplace=True)
    exact = abs(Statevector(qc).data[0]) ** 2          # amplitude of |0...0>
    qc.measure_all()
    return qc, exact


def build(ns, L, pairs, seed):
    rng = np.random.default_rng(seed)
    jobs = []
    for n in ns:
        for i in range(pairs):
            a, b = rng.uniform(0, 2 * math.pi, n), rng.uniform(0, 2 * math.pi, n)
            tq, tk = rng.uniform(0, 2 * math.pi, n * L), rng.uniform(0, 2 * math.pi, n * L)
            qc, exact = fidelity_circuit(a, b, tq, tk, n, L)
            jobs.append(dict(n=n, pair=i, circuit=qc, exact=exact))
    return jobs


def run_sampler(jobs, shots, backend_name, dry_run):
    if dry_run:
        from qiskit.primitives import StatevectorSampler
        sampler, circuits, backend = StatevectorSampler(seed=0), [j["circuit"] for j in jobs], "statevector"
    else:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        service = QiskitRuntimeService(channel="ibm_quantum_platform")   # reads QISKIT_IBM_TOKEN
        be = (service.least_busy(operational=True, simulator=False) if backend_name == "least_busy"
              else service.backend(backend_name))
        pm = generate_preset_pass_manager(optimization_level=3, backend=be)
        circuits = [pm.run(j["circuit"]) for j in jobs]
        sampler, backend = SamplerV2(mode=be), be.name
        print(f"Submitting {len(circuits)} circuits x {shots} shots to {backend} ...")
    result = sampler.run(circuits, shots=shots).result()
    for j, r in zip(jobs, result):
        counts = r.data.meas.get_counts()
        j["measured"] = counts.get("0" * j["n"], 0) / shots
    return backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="least_busy")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--qubits", default="2,3,4,5,6,8")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    ns = [int(n) for n in args.qubits.split(",")]
    jobs = build(ns, args.layers, args.pairs, args.seed)
    backend = run_sampler(jobs, args.shots, args.backend, args.dry_run)

    out = OUT.replace(".csv", "_dryrun.csv") if args.dry_run else OUT
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["backend", "n", "L", "pair", "shots", "exact", "measured"])
        w.writeheader()
        for j in jobs:
            w.writerow(dict(backend=backend, n=j["n"], L=args.layers, pair=j["pair"], shots=args.shots,
                            exact=j["exact"], measured=j["measured"]))
    print(f"\nbackend: {backend}   (written to {os.path.abspath(out)})")
    # Device noise also pushes P(0...0) towards 2^-n, so the mean alone cannot separate signal from
    # decoherence; the per-circuit correlation between measured and exact scores can.
    print(" n   mean exact   mean measured   2^-n     corr(measured, exact)")
    for n in ns:
        e = np.array([j["exact"] for j in jobs if j["n"] == n])
        m = np.array([j["measured"] for j in jobs if j["n"] == n])
        r = np.corrcoef(e, m)[0, 1] if e.std() > 0 and m.std() > 0 else float("nan")
        print(f"{n:2d}   {e.mean():10.4f}   {m.mean():13.4f}   {2.0 ** -n:.4f}   {r:8.3f}")


if __name__ == "__main__":
    main()
