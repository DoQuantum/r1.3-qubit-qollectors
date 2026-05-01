# Qubit Qollectors — Qubit-to-Ququart Circuit Mapping

A pipeline for compiling qubit-based quantum circuits into ququart-based circuits using graph partitioning to minimize inter-ququart gate overhead.

---

## Background

A **ququart** is a 4-dimensional quantum unit equivalent to two qubits. When mapping a qubit circuit to ququart hardware, 2-qubit gates that operate *within* the same ququart are cheap (single-ququart gates), while 2-qubit gates that operate *across* ququarts are expensive (two-ququart gates) due to significantly slower gate speeds. Single-qubit gates become single-ququart gates and are negligible in cost.

The goal of this project is to find the qubit-to-ququart assignment that minimizes the number of two-ququart gates in the compiled circuit.

---

## Approach

### 1. Circuit to Interaction Graph

The input qubit circuit is transformed into a weighted interaction graph:

- **Nodes** — individual qubits
- **Edges** — pairs of qubits that share at least one 2-qubit gate
- **Edge weight** — total number of 2-qubit gates acting on that pair

### 2. Graph Partitioning via Kernighan-Lin

The interaction graph is bisected using the **Kernighan-Lin (KL) algorithm**, which produces two balanced partitions of qubits. Each partition becomes one ququart. KL iteratively swaps qubits between partitions to reduce the weighted edge cut — the sum of gate weights crossing the partition boundary.

This directly minimizes the number of two-ququart gates in the output circuit.

### 3. Ququart Circuit Generation

Using the KL partitioning:

- **2-qubit gates within the same partition** → compiled as single-ququart gates
- **2-qubit gates across partitions** → compiled as two-ququart gates
- **Single-qubit gates** → compiled as single-ququart gates

---

## Complexity

| Strategy | Complexity | Notes |
|---|---|---|
| Brute force | O(n!) | Optimal, but only tractable up to ~4 qubits |
| Kernighan-Lin | O(n² log n) | Heuristic, scales to large circuits |

The brute force approach enumerates all possible qubit pairings and selects the one with minimum cross-partition cost. It is kept in the codebase as a correctness baseline for small circuits.

---

## Cost Function

The cost function counts the total weight of edges that cross the ququart boundary:

```
cost = sum of edge weights (q_i, q_j) where q_i and q_j are in different ququarts
```

Minimizing this cost minimizes the number of expensive two-ququart gates in the compiled output.

---

## Dependencies

- [QuTiP](https://qutip.org/) — quantum state and operator representation
- [NetworkX](https://networkx.org/) — graph construction and KL bisection
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — numerical computation
- [Matplotlib](https://matplotlib.org/) — partition visualization

Install dependencies:

```bash
pip install qutip networkx numpy scipy matplotlib
```

---

## Repository Structure

```
.
├── qutip_tutorial.ipynb   # Algorithmic exploration and experiments
└── qudit/                 # Python virtual environment
```

The notebook documents the full development arc: ququart state construction, brute force baseline, KL partitioning, and a three-algorithm comparison (Kernighan-Lin, Spectral, Fiduccia-Mattheyses).

---

## Algorithm Comparison

Three graph partitioning algorithms were evaluated:

| Algorithm | Strategy | Notes |
|---|---|---|
| **Kernighan-Lin** | Iterative swap-based bisection | Best overall performance, selected for pipeline |
| **Spectral** | Fiedler vector of graph Laplacian | Mathematically principled, sensitive to graph structure |
| **Fiduccia-Mattheyses** | Greedy node-move heuristic | Fast but less consistent |
