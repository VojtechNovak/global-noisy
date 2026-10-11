"""Optional independent Qiskit cross-check; not a primary runtime dependency.

Install qiskit locally if desired: python -m pip install qiskit
"""
import numpy as np
from .models import make_hamiltonian, Problem


def main():
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, SparsePauliOp
    except ImportError as exc:
        raise SystemExit("Optional validation requires qiskit; the NumPy study runs without it.") from exc
    for family in ["glass", "mixed23", "maxcut"]:
        p = Problem(make_hamiltonian(4, family, 19), "qaoa", depth=6, blocks=3)
        x = np.random.default_rng(91).uniform(0, 1, p.dim)
        angles = p.angles(x)
        qc = QuantumCircuit(p.n); qc.h(range(p.n))
        for layer in range(p.depth):
            b = layer % p.blocks
            for coefficient, support in zip(p.h.coefficients, p.h.supports):
                label = "".join("Z" if q in support else "I" for q in reversed(range(p.n)))
                op = SparsePauliOp(label, coeffs=[coefficient])
                from qiskit.circuit.library import PauliEvolutionGate
                qc.append(PauliEvolutionGate(op, time=angles[2 * b]), range(p.n))
            for q in range(p.n):
                qc.rx(2 * angles[2 * b + 1], q)
        qiskit_state = Statevector.from_instruction(qc).data
        np.testing.assert_allclose(qiskit_state, p.state(x), atol=1e-11)
        print(family, "NumPy / Qiskit statevectors agree")


if __name__ == "__main__":
    main()
