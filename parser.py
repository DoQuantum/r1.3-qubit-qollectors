"""
Circuit Parser for Qubit-Qollector Graph Builder
=================================================
Parses quantum circuits from supported frameworks into a weighted interaction
graph (nodes = qubits, edges = weighted multi-qubit interaction count) that
can be passed directly to the three partitioning algorithms in qutip_tutorial.

Supported input formats:
  - Qiskit QuantumCircuit object
  - OpenQASM 2.0 string

Usage:
    from parser import parse_qiskit, parse_qasm, build_interaction_graph
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    qc.h(0); qc.cx(0, 1); qc.cz(2, 3); qc.swap(1, 2)

    G = parse_qiskit(qc)
    # G is a networkx.Graph ready for kernighan_lin_partition,
    # spectral_partition, or fiduccia_mattheyses_partition
"""

import builtins
import warnings
from collections import defaultdict

import networkx as nx

# ---------------------------------------------------------------------------
# Gate weights — mirror the values used in qutip_tutorial.ipynb.
# Any 2-qubit gate not listed here gets weight DEFAULT_WEIGHT.
# ---------------------------------------------------------------------------
GATE_WEIGHTS: dict[str, int] = {
    # Canonical names
    "cx":    3,   # CNOT
    "cnot":  3,
    "cz":    2,
    "swap":  5,
    "iswap": 4,
    # Common aliases
    "ch":    2,   # Controlled-H (treat like CZ weight)
    "cy":    3,   # Controlled-Y (treat like CNOT weight)
    "cp":    2,   # Controlled-Phase
    "crx":   3,
    "cry":   3,
    "crz":   3,
    "csx":   3,   # Controlled-SX
    "dcx":   3,   # Double-CNOT
    "ecr":   3,   # Echoed cross-resonance
    "rzx":   3,
    "rxx":   3,
    "ryy":   3,
    "rzz":   2,
}

DEFAULT_WEIGHT: int = 1  # fallback for unrecognised 2-qubit gates


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _gate_weight(gate_name: str) -> int:
    """Return the interaction weight for a named gate."""
    return GATE_WEIGHTS.get(gate_name.lower(), DEFAULT_WEIGHT)


def build_interaction_graph(
    num_qubits: int,
    interactions: dict[tuple[int, int], dict[str, int]],
) -> nx.Graph:
    """
    Build a weighted NetworkX graph from raw interaction counts.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits in the circuit.
    interactions : dict mapping (q_i, q_j) -> {gate_name: count}
        q_i < q_j always (canonical ordering).

    Returns
    -------
    nx.Graph
        Nodes are labelled 'q0', 'q1', …, 'q{n-1}'.
        Each edge weight is the sum of (GATE_WEIGHTS[gate] * count) for all
        multi-qubit gates between those two qubits.
    """
    qubit_labels = [f"q{i}" for i in range(num_qubits)]

    G = nx.Graph()
    G.add_nodes_from(qubit_labels)

    for (qi, qj), gate_dict in interactions.items():
        weight = builtins.sum(_gate_weight(g) * c for g, c in gate_dict.items())
        if weight > 0:
            G.add_edge(qubit_labels[qi], qubit_labels[qj], weight=weight)

    return G


# ---------------------------------------------------------------------------
# Qiskit parser
# ---------------------------------------------------------------------------

def parse_qiskit(circuit) -> nx.Graph:
    """
    Parse a Qiskit QuantumCircuit into a weighted interaction graph.

    Parameters
    ----------
    circuit : qiskit.circuit.QuantumCircuit
        Any Qiskit circuit. Multi-qubit gates contribute to edge weights;
        single-qubit gates are ignored. 3+-qubit gates (e.g. Toffoli) are
        decomposed into all unique qubit pairs they touch.

    Returns
    -------
    nx.Graph
        Weighted interaction graph ready for the partitioning algorithms.
    """
    try:
        from qiskit.circuit import QuantumCircuit  # noqa: F401 — import check
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required: pip install qiskit"
        ) from exc

    num_qubits = circuit.num_qubits
    interactions: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for instruction in circuit.data:
        op = instruction.operation
        qargs = instruction.qubits

        if op.num_qubits < 2:
            continue  # single-qubit gate — no interaction to record

        gate_name = op.name.lower()
        qubit_indices = [circuit.find_bit(q).index for q in qargs]

        if op.num_qubits == 2:
            qi, qj = sorted(qubit_indices)
            interactions[(qi, qj)][gate_name] += 1
        else:
            # 3+-qubit gate: record every unique pair it touches
            warnings.warn(
                f"Gate '{op.name}' acts on {op.num_qubits} qubits. "
                "Decomposing into all qubit pairs for interaction counting.",
                UserWarning,
                stacklevel=2,
            )
            from itertools import combinations
            for qi, qj in combinations(sorted(qubit_indices), 2):
                interactions[(qi, qj)][gate_name] += 1

    return build_interaction_graph(num_qubits, interactions)


# ---------------------------------------------------------------------------
# OpenQASM 2.0 parser
# ---------------------------------------------------------------------------

def parse_qasm(qasm_str: str) -> nx.Graph:
    """
    Parse an OpenQASM 2.0 string into a weighted interaction graph.

    Most quantum frameworks can export to QASM 2.0:
      - Qiskit:    qiskit.qasm2.dumps(circuit)
      - Cirq:      cirq.qasm(circuit)
      - PyQuil:    program.out() (converts to Quil, then QASM via transpile)

    Parameters
    ----------
    qasm_str : str
        A valid OpenQASM 2.0 program as a string.

    Returns
    -------
    nx.Graph
        Weighted interaction graph ready for the partitioning algorithms.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError(
            "Qiskit is required to parse QASM: pip install qiskit"
        ) from exc

    import qiskit.qasm2 as qasm2
    circuit = qasm2.loads(qasm_str, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    return parse_qiskit(circuit)


# ---------------------------------------------------------------------------
# Convenience: show graph summary
# ---------------------------------------------------------------------------

def print_graph_summary(G: nx.Graph) -> None:
    """Print a human-readable summary of the interaction graph."""
    print(f"Qubits  : {sorted(G.nodes())}")
    print(f"Edges   : {G.number_of_edges()}")
    print("Interactions:")
    for u, v, data in sorted(G.edges(data=True)):
        print(f"  {u} — {v}  weight={data['weight']}")
    total = builtins.sum(d["weight"] for _, _, d in G.edges(data=True))
    print(f"Total weighted interactions: {total}")


# ---------------------------------------------------------------------------
# Demo — run `python parser.py` to see a quick example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        print("Install Qiskit to run the demo:  pip install qiskit")
        raise SystemExit(1)

    # Build a sample 4-qubit Qiskit circuit
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    # Multi-qubit interactions
    qc.cx(0, 1)   # CNOT → weight 3
    qc.cx(0, 1)   # second CNOT on same pair → weight 3 again
    qc.cx(0, 1)   # third → weight 3
    qc.cx(0, 1)   # fourth → edge (q0,q1) total: 4×3 = 12
    qc.cz(2, 3)   # CZ → weight 2
    qc.cz(2, 3)   # second CZ → edge (q2,q3) total: 2×2 = 4 (but notebook uses raw counts)
    qc.cz(2, 3)   # third
    qc.swap(0, 3) # SWAP → weight 5 (edge q0,q3)
    qc.swap(0, 3)
    qc.swap(0, 3)
    qc.swap(0, 3)
    qc.swap(0, 3)
    qc.swap(0, 3)  # 6 SWAPs → edge (q0,q3): 6×5 = 30

    print("=== Sample Qiskit Circuit ===")
    print(qc.draw(output="text"))
    print()

    G = parse_qiskit(qc)

    print("=== Interaction Graph ===")
    print_graph_summary(G)
    print()

    # Show it's ready for the partitioning algorithms
    print("=== Kernighan-Lin Partition ===")
    from networkx.algorithms.community import kernighan_lin_bisection
    import random
    nodes = list(G.nodes())
    random.shuffle(nodes)
    half = len(nodes) // 2
    A, B = kernighan_lin_bisection(
        G, partition=(set(nodes[:half]), set(nodes[half:])), weight="weight"
    )
    print(f"  Ququart A: {sorted(A)}")
    print(f"  Ququart B: {sorted(B)}")
    cut = builtins.sum(
        d["weight"]
        for u, v, d in G.edges(data=True)
        if (u in A and v in B) or (u in B and v in A)
    )
    print(f"  Cross-ququart cost: {cut}")
