"""NumPy simulators. Qubit 0 is the LEAST significant bit, as in Qiskit.

Optimization coordinates u are in [0,1]^d; the objective is normalized by
the full diagonal spectrum, but QAOA evolution uses the RAW Hamiltonian.
All QAOA coefficients are integers, making the declared periods valid.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
import numpy as np


def stable_seed(*parts):
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def cycle_matching(n):
    if n < 4 or n % 2:
        raise ValueError("The ring-plus-matching graph requires even n >= 4.")
    return sorted({tuple(sorted((i, (i + 1) % n))) for i in range(n)} |
                  {(i, i + n // 2) for i in range(n // 2)})


@dataclass
class DiagonalHamiltonian:
    n: int
    supports: list
    coefficients: np.ndarray

    def __post_init__(self):
        self.supports = [tuple(map(int, s)) for s in self.supports]
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        if len(self.supports) != len(self.coefficients):
            raise ValueError("One coefficient per Pauli support is required.")
        if any(len(set(s)) != len(s) or any(q < 0 or q >= self.n for q in s)
               for s in self.supports):
            raise ValueError("Invalid Pauli support.")
        self.masks = np.array([sum(1 << q for q in s) for s in self.supports], dtype=np.uint32)
        self.bits = ((np.arange(1 << self.n, dtype=np.uint32)[:, None] >>
                      np.arange(self.n)) & 1).astype(np.int8)
        self.z = 1 - 2 * self.bits
        self.diagonal = np.zeros(1 << self.n)
        for c, s in zip(self.coefficients, self.supports):
            self.diagonal += c * np.prod(self.z[:, s], axis=1)
        self.e0 = float(self.diagonal.min())
        self.span = float(np.ptp(self.diagonal))
        if self.span <= 0:
            raise ValueError("A nonconstant Hamiltonian is required.")
        self.normalized = (self.diagonal - self.e0) / self.span
        squared = {}
        for a, ca in zip(self.masks, self.coefficients):
            for b, cb in zip(self.masks, self.coefficients):
                mask = int(a ^ b)
                squared[mask] = squared.get(mask, 0.0) + ca * cb
        self.square_supports = [tuple(q for q in range(self.n) if m & (1 << q))
                                for m in squared]
        self.square_coefficients = np.array(list(squared.values()))
        self.square_incidence = np.array([[q in s for q in range(self.n)]
                                          for s in self.square_supports])

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.metadata(), sort_keys=True).encode()).hexdigest()

    def metadata(self):
        return dict(n=self.n, supports=self.supports, coefficients=self.coefficients.tolist(),
                    e0=self.e0, span=self.span)


def make_hamiltonian(n, family, seed, frustration=0.5, mixed_strength=1):
    rng = np.random.default_rng(seed)
    if family == "separable":
        return DiagonalHamiltonian(n, [(i,) for i in range(n)], -np.ones(n))
    edges = cycle_matching(n)
    if family == "maxcut":
        return DiagonalHamiltonian(n, edges, np.ones(len(edges)))
    if family not in {"q1", "glass", "mixed23"}:
        raise ValueError(f"Unknown Hamiltonian family: {family}")
    signs = np.where(rng.random(len(edges)) < frustration, 1.0, -1.0)
    supports, coeff = list(edges), list(signs)
    if family == "q1":
        supports += [(i,) for i in range(n)]
        coeff += list(rng.uniform(-0.2, 0.2, n))
    if family == "mixed23":
        if int(mixed_strength) != mixed_strength or mixed_strength < 0:
            raise ValueError("Integer mixed strengths preserve the QAOA gamma period.")
        triples = list(combinations(range(n), 3))
        ids = rng.choice(len(triples), min(n, len(triples)), replace=False)
        supports += [triples[i] for i in ids]
        coeff += list(mixed_strength * rng.choice([-1.0, 1.0], len(ids)))
    return DiagonalHamiltonian(n, supports, np.array(coeff))


class Problem:
    def __init__(self, hamiltonian, ansatz="product", depth=3, blocks=None,
                 shear=0, problem_id="problem"):
        self.h = hamiltonian
        self.n, self.ansatz = self.h.n, ansatz
        self.depth = int(depth)
        self.blocks = int(self.depth if blocks is None else blocks)
        if self.ansatz not in {"product", "qaoa"}:
            raise ValueError("Ansatz must be product or qaoa.")
        if self.depth < 1 or not 1 <= self.blocks <= self.depth:
            raise ValueError("Require depth >= blocks >= 1.")
        self.dim = self.n if ansatz == "product" else 2 * self.blocks
        self.shear = int(shear)
        self.U = np.eye(self.dim, dtype=int)
        # An integer unimodular torus shear is bijective. A real rotation of a
        # cube would change the feasible domain and is intentionally not used.
        if self.shear:
            self.U[np.arange(self.dim - 1), np.arange(1, self.dim)] = self.shear
        if ansatz == "qaoa" and not np.allclose(self.h.coefficients,
                                                np.rint(self.h.coefficients)):
            raise ValueError("QAOA torus requires integer raw Hamiltonian coefficients.")
        self.problem_id = problem_id

    def coordinates(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape != (self.dim,) or not np.all(np.isfinite(u)):
            raise ValueError(f"Expected finite shape ({self.dim},).")
        return (self.U @ u) % 1.0

    def angles(self, u):
        x = 2 * np.pi * (self.coordinates(u) - 0.5)
        if self.ansatz == "qaoa":
            x[1::2] *= 0.5
        return x

    def state(self, u):
        angles = self.angles(u)
        if self.ansatz == "product":
            psi = np.ones(1 << self.n, dtype=complex)
            for q in range(self.n):
                psi *= np.where(self.h.bits[:, q], np.sin(angles[q] / 2),
                                np.cos(angles[q] / 2))
            return psi
        psi = np.full(1 << self.n, 1 / np.sqrt(1 << self.n), dtype=complex)
        phases = [np.exp(-1j * angles[2 * b] * self.h.diagonal)
                  for b in range(self.blocks)]
        for layer in range(self.depth):
            b = layer % self.blocks
            psi *= phases[b]
            c, s = np.cos(angles[2 * b + 1]), np.sin(angles[2 * b + 1])
            for q in range(self.n):
                # RX(2 beta) = exp(-i beta X); no Kronecker matrices.
                v = psi.reshape(-1, 2, 1 << q)
                a, d = v[:, 0, :].copy(), v[:, 1, :].copy()
                v[:, 0, :] = c * a - 1j * s * d
                v[:, 1, :] = c * d - 1j * s * a
        return psi

    def moments(self, u, need_variance=True):
        if self.ansatz == "product":
            c = np.cos(self.angles(u))
            raw = sum(a * np.prod(c[list(s)])
                      for a, s in zip(self.h.coefficients, self.h.supports))
            variance = 0.0
            if need_variance:
                products = np.prod(np.where(self.h.square_incidence, c, 1.0), axis=1)
                second = float(np.sum(self.h.square_coefficients * products))
                variance = max(0.0, second - raw * raw) / self.h.span ** 2
            return float((raw - self.h.e0) / self.h.span), float(variance)
        prob = np.abs(self.state(u)) ** 2
        mean = float(np.sum(prob * self.h.normalized))
        var = max(0.0, float(np.sum(prob * self.h.normalized ** 2)) - mean ** 2)
        return mean, var if need_variance else 0.0

    def objective(self, u):
        return self.moments(u, False)[0]

    def sample_mean(self, u, shots, rng):
        if int(shots) != shots or shots < 1:
            raise ValueError("Sampling requires a positive integer shot count.")
        if self.ansatz == "product":
            p1 = (1 - np.cos(self.angles(u))) / 2
            bits = rng.random((int(shots), self.n)) < p1
            indices = bits.astype(np.uint32) @ (1 << np.arange(self.n, dtype=np.uint32))
            return float(np.mean(self.h.normalized[indices]))
        prob = np.abs(self.state(u)) ** 2
        prob /= prob.sum()
        ids = rng.choice(len(prob), size=int(shots), p=prob)
        return float(np.mean(self.h.normalized[ids]))

    def gradient(self, u, step=2e-5):
        if self.ansatz == "qaoa":
            eye = np.eye(self.dim) * step
            return np.array([(self.objective(u + e) - self.objective(u - e)) /
                             (2 * step) for e in eye])
        angles = self.angles(u)
        c, s = np.cos(angles), np.sin(angles)
        g = np.zeros(self.n)
        for a, support in zip(self.h.coefficients, self.h.supports):
            for q in support:
                g[q] -= a * s[q] * np.prod([c[j] for j in support if j != q])
        return 2 * np.pi * (self.U.T @ g) / self.h.span

    def hessian(self, u, step=2e-4):
        if self.ansatz == "qaoa":
            H = np.zeros((self.dim, self.dim))
            center = self.objective(u)
            eye = np.eye(self.dim) * step
            for i in range(self.dim):
                H[i, i] = (self.objective(u + eye[i]) - 2 * center +
                           self.objective(u - eye[i])) / step ** 2
                for j in range(i):
                    value = (self.objective(u + eye[i] + eye[j]) -
                             self.objective(u + eye[i] - eye[j]) -
                             self.objective(u - eye[i] + eye[j]) +
                             self.objective(u - eye[i] - eye[j])) / (4 * step ** 2)
                    H[i, j] = H[j, i] = value
            return H
        c, s = np.cos(self.angles(u)), np.sin(self.angles(u))
        H = np.zeros((self.n, self.n))
        for a, support in zip(self.h.coefficients, self.h.supports):
            v = a * np.prod(c[list(support)])
            for q in support:
                H[q, q] -= v
            for i, q in enumerate(support):
                for r in support[:i]:
                    vqr = a * s[q] * s[r] * np.prod([c[j] for j in support if j not in (q, r)])
                    H[q, r] += vqr
                    H[r, q] += vqr
        return (2 * np.pi) ** 2 * (self.U.T @ H @ self.U) / self.h.span

    def metadata(self):
        return dict(problem_id=self.problem_id, hamiltonian=self.h.metadata(),
                    fingerprint=self.h.fingerprint(), ansatz=self.ansatz,
                    depth=self.depth, blocks=self.blocks, dim=self.dim,
                    shear=self.shear, periods="unit torus; raw integer QAOA generator")


def build_problem(spec, n, instance_seed):
    family = spec["family"]
    h = make_hamiltonian(n, family, instance_seed,
                         spec.get("frustration", 0.5), spec.get("mixed_strength", 1))
    ident = f'{spec["name"]}_n{n}_seed{instance_seed}'
    return Problem(h, spec.get("ansatz", "product"), spec.get("depth", 3),
                   spec.get("blocks"), spec.get("shear", 0), ident)
