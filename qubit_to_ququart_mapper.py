"""
Qubit-to-QuQuart Circuit Mapper

This program converts qubit circuits (2-level systems) into qu-quart circuits
(4-level systems) while minimizing cross-qu-quart gate operations using QuTiP.

Author: Qubit Qollectors
Date: 2026-01-22
"""

import numpy as np
import qutip as qt
from typing import List, Tuple, Dict
from itertools import combinations


# --- Gate constants and identification ---

SWAP_2Q = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype = complex)

KNOWN_GATES = {
    "I": np.eye(2, dtype = complex),
    "X": np.array([[0, 1], [1, 0]], dtype = complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype = complex),
    "Z": np.array([[1, 0], [0, -1]], dtype = complex),
    "H": (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype = complex),
    "S": np.array([[1, 0], [0, 1j]], dtype = complex),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype = complex),
    "CNOT": np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0],
        [0, 0, 0, 1], 
        [0, 0, 1, 0]
    ], dtype = complex),
    "CZ": np.diag([1, 1, 1, -1]).astype(complex),
    "CH": np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
        [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]
    ], dtype=complex),
    "CY": np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0]
    ], dtype = complex),
    "ISWAP": np.array([
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex),
    "SWAP": SWAP_2Q.copy()
}


def identify_gate(gate_matrix: np.ndarray) -> str:
    """
    Identify a gate matrix by comparing against known gates.
    For 4x4 matrices, also recognizes G⊗I and I⊗G tensor product structures.
    Returns gate name string, or 'U(NxN)' for unrecognized matrices.
    """
    # Exact match against known gates
    for name, known in KNOWN_GATES.items():
        if gate_matrix.shape == known.shape and np.allclose(gate_matrix, known):
            return name

    n = gate_matrix.shape[0]

    # For 4x4 ququart gates, check for single-qubit-plus-identity tensor structure
    if gate_matrix.shape == (4, 4):
        # Check G⊗I₂: M[i,j] = G[i//2, j//2] if i%2==j%2, else 0
        # Reconstruct G from the even-indexed rows/cols
        G_candidate = np.array([[gate_matrix[0, 0], gate_matrix[0, 2]],
                                 [gate_matrix[2, 0], gate_matrix[2, 2]]], dtype=complex)
        if np.allclose(gate_matrix, np.kron(G_candidate, np.eye(2))):
            for g_name, known in KNOWN_GATES.items():
                if known.shape == (2, 2) and np.allclose(G_candidate, known):
                    return f"{g_name} \u2297 I"

        # Check I₂⊗G: M is block-diagonal with G in each 2x2 block
        # Reconstruct G from the top-left 2x2 block
        G_candidate = gate_matrix[:2, :2].copy()
        if np.allclose(gate_matrix, np.kron(np.eye(2), G_candidate)):
            for g_name, known in KNOWN_GATES.items():
                if known.shape == (2, 2) and np.allclose(G_candidate, known):
                    return f"I \u2297 {g_name}"

    return f"U({n}x{n})"


class QubitCircuit:
    """
    Represents a qubit circuit as a list of layers.
    Each layer contains gates that act on different qubits simultaneously.
    """
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.layers = []

    def add_layer(self, gates: List[Tuple[np.ndarray, List[int]]]):
        """
        Add a layer of gates to the circuit.

        Parameters:
        -----------
        gates : List[Tuple[np.ndarray, List[int]]]
            List of (gate_matrix, [qubit_indices]) tuples
        """
        self.layers.append(gates)

    def __repr__(self):
        return f"QubitCircuit({self.num_qubits} qubits, {len(self.layers)} layers)"


class QuQuartCircuit:
    """
    Represents a qu-quart circuit as a list of layers.
    Each layer contains gates that act on qu-quarts (4-level systems).
    """
    def __init__(self, num_ququarts: int, mapping: List[Tuple[int, int]]):
        self.num_ququarts = num_ququarts
        self.mapping = mapping  # Which qubits are paired into each qu-quart
        self.layers = []
        self.cross_ququart_gate_count = 0

    def add_layer(self, gates: List[Tuple[np.ndarray, List[int]]]):
        """
        Add a layer of gates to the qu-quart circuit.

        Parameters:
        -----------
        gates : List[Tuple[np.ndarray, List[int]]]
            List of (gate_matrix, [ququart_indices]) tuples
        """
        self.layers.append(gates)
        for gate_matrix, ququart_indices in gates:
            if len(ququart_indices) > 1:
                self.cross_ququart_gate_count += 1

    def __repr__(self):
        return (f"QuQuartCircuit({self.num_ququarts} qu-quarts, {len(self.layers)} layers, "
                f"{self.cross_ququart_gate_count} cross-qu-quart gates)")


def best_mapping_optimization(qubit_circuit: QubitCircuit) -> List[Tuple[int, int]]:
    """
    Best Mapping Optimization (BMO) Function

    Finds the optimal qubit pairing to minimize cross-qu-quart gates by analyzing
    the frequency of 2-qubit gates between different qubit pairs.

    Parameters:
    -----------
    qubit_circuit : QubitCircuit
        The input qubit circuit to analyze

    Returns:
    --------
    List[Tuple[int, int]]
        Optimal pairing of qubits, e.g., [(0,1), (2,3)]

    Algorithm:
    ----------
    1. Build weighted graph with qubits as nodes
    2. Edge weights = count of 2-qubit gates between those qubits
    3. Select pairing that maximizes within-qu-quart gates
    """
    num_qubits = qubit_circuit.num_qubits

    if num_qubits != 4:
        raise ValueError("BMO currently only supports 4-qubit systems")

    # Step 1: Build interaction graph (count 2-qubit gates between each pair)
    interaction_counts = {}
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            interaction_counts[(i, j)] = 0

    # Count 2-qubit gates in all layers
    for layer in qubit_circuit.layers:
        for gate_matrix, qubit_indices in layer:
            if len(qubit_indices) == 2:
                q1, q2 = sorted(qubit_indices)
                interaction_counts[(q1, q2)] += 1

    # Step 2: Evaluate all possible pairings for 4 qubits
    # Valid pairings: (0,1)+(2,3), (0,2)+(1,3), (0,3)+(1,2)
    possible_pairings = [
        [(0, 1), (2, 3)],  # Option A
        [(0, 2), (1, 3)],  # Option B
        [(0, 3), (1, 2)]   # Option C
    ]

    # Step 3: Calculate cost for each pairing
    # Cost = number of 2-qubit gates that cross qu-quart boundaries
    best_pairing = None
    min_cross_gates = float('inf')

    for pairing in possible_pairings:
        cross_gate_count = 0

        # Count gates that connect qubits in different qu-quarts
        for (q1, q2), count in interaction_counts.items():
            # Check if q1 and q2 are in different qu-quarts
            q1_ququart = 0 if q1 in pairing[0] else 1
            q2_ququart = 0 if q2 in pairing[0] else 1

            if q1_ququart != q2_ququart:
                cross_gate_count += count

        # Select pairing with minimum cross-qu-quart gates
        if cross_gate_count < min_cross_gates:
            min_cross_gates = cross_gate_count
            best_pairing = pairing

    print(f"BMO Analysis:")
    print(f"  Interaction counts: {interaction_counts}\n")
    print(f"  Most optimal pairing: {best_pairing}")
    print(f"  Cross-qu-quart gates: {min_cross_gates}")

    return best_pairing


def get_ququart_index(qubit_idx: int, mapping: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Determine which qu-quart a qubit belongs to and its position within that qu-quart.

    Parameters:
    -----------
    qubit_idx : int
        The qubit index
    mapping : List[Tuple[int, int]]
        The qubit-to-ququart mapping

    Returns:
    --------
    Tuple[int, int]
        (ququart_index, position_in_ququart)
        position_in_ququart: 0 = lower level, 1 = upper level
    """
    for ququart_idx, (q1, q2) in enumerate(mapping):
        if qubit_idx == q1:
            return (ququart_idx, 0)
        elif qubit_idx == q2:
            return (ququart_idx, 1)
    raise ValueError(f"Qubit {qubit_idx} not found in mapping")


def build_basis_permutation(mapping: List[Tuple[int, int]]) -> np.ndarray:
    """
    Build the 16x16 permutation matrix P that changes basis from the
    standard 4-qubit computational basis (index = 8*q0+4*q1+2*q2+q3)
    to the ququart-level basis (index = 4*QQ_A + QQ_B).

    Ququart level encoding:  level = pos0_val + 2 * pos1_val
      pos-0 qubit contributes 1 (LSB of ququart level)
      pos-1 qubit contributes 2 (MSB of ququart level)

    For mapping [(a,b),(c,d)]:
        QQ_A = q_a + 2*q_b   (a is pos-0, b is pos-1 of QQ_A)
        QQ_B = q_c + 2*q_d   (c is pos-0, d is pos-1 of QQ_B)
        ququart index = 4*QQ_A + QQ_B = 4*q_a + 8*q_b + q_c + 2*q_d

    Usage:
        P = build_basis_permutation(mapping)
        M_ququart = P @ M_standard @ P.T     (P.T == P^{-1} since P is a permutation)
        M_standard = P.T @ M_ququart @ P
    """
    (a, b), (c, d) = mapping
    P = np.zeros((16, 16), dtype=complex)
    for s in range(16):
        q = [(s >> (3 - k)) & 1 for k in range(4)]   # bit of qubit k
        QQ_A = q[a] + 2 * q[b]                        # pos0_val + 2*pos1_val
        QQ_B = q[c] + 2 * q[d]
        qq_idx = 4 * QQ_A + QQ_B
        P[qq_idx, s] = 1.0
    return P


def standard_to_ququart_basis(M_std: np.ndarray,
                               mapping: List[Tuple[int, int]]) -> np.ndarray:
    """
    Convert a 16x16 matrix from the standard 4-qubit basis to the ququart-level basis.

    Parameters:
    -----------
    M_std : np.ndarray
        16x16 matrix in the standard |q0 q1 q2 q3⟩ basis
    mapping : List[Tuple[int, int]]
        The qubit-to-ququart mapping

    Returns:
    --------
    np.ndarray
        16x16 matrix in the ququart-level |QQ_A, QQ_B⟩ basis
    """
    P = build_basis_permutation(mapping)
    return P @ M_std @ P.T


#Instead of constructing the 16x16 matrix from scratch, use tensoring and swaps within ququarts
#
def embed_cross_ququart_gate(gate_matrix: np.ndarray, qubit_a: int, qubit_b: int) -> qt.Qobj:
    """
    Embed a 2-qubit gate into the 16-dimensional space of two ququarts.

    The 16-dim basis is the standard 4-qubit computational basis:
        index = 8*q0 + 4*q1 + 2*q2 + q3
    so bits[k] = (index >> (3-k)) & 1 gives the bit for qubit k.

    The gate acts on qubit qubit_a (first/control) and qubit qubit_b
    (second/target) in this standard basis. The other two qubits are
    spectators (identity).

    Parameters:
    -----------
    gate_matrix : np.ndarray
        4x4 unitary gate matrix for the 2-qubit gate
    qubit_a : int
        Standard qubit index (0-3) for the gate's first qubit
    qubit_b : int
        Standard qubit index (0-3) for the gate's second qubit

    Returns:
    --------
    qt.Qobj
        16x16 matrix acting on the joint two-ququart space, dims=[[4,4],[4,4]]
    """
    G = np.array(gate_matrix, dtype=complex)
    M = np.zeros((16, 16), dtype=complex)

    target_a = qubit_a   # bit position in the 4-qubit standard basis
    target_b = qubit_b   # bit position in the 4-qubit standard basis

    for i_out in range(16):
        for i_in in range(16):
            # Extract 4-bit representation: bits[k] = bit value of qubit k
            bits_out = [(i_out >> (3 - k)) & 1 for k in range(4)]
            bits_in = [(i_in >> (3 - k)) & 1 for k in range(4)]

            # Spectator qubits must have matching bits
            spectators_match = all(
                bits_out[k] == bits_in[k]
                for k in range(4)
                if k != target_a and k != target_b
            )
            if not spectators_match:
                continue

            # Map active qubit bits to the gate's row/column indices
            g_row = bits_out[target_a] * 2 + bits_out[target_b]
            g_col = bits_in[target_a] * 2 + bits_in[target_b]
            M[i_out, i_in] = G[g_row, g_col]

    return qt.Qobj(M, dims=[[4, 4], [4, 4]])


def qubit_gate_to_ququart(gate_matrix: np.ndarray,
                          qubit_indices: List[int],
                          mapping: List[Tuple[int, int]]) -> Tuple[qt.Qobj, List[int], bool]:
    """ 
    Convert a qubit gate to its qu-quart representation.

    Parameters:
    -----------
    gate_matrix : np.ndarray
        The gate matrix in qubit representation
    qubit_indices : List[int]
        Which qubits the gate acts on
    mapping : List[Tuple[int, int]]
        The qubit-to-ququart mapping

    Returns:
    --------
    Tuple[qt.Qobj, List[int], bool]
        (ququart_gate_matrix, ququart_indices, is_cross_ququart)
    """
    gate_qobj = qt.Qobj(gate_matrix)

    if len(qubit_indices) == 1:
        # Single-qubit gate
        qubit_idx = qubit_indices[0]
        ququart_idx, pos = get_ququart_index(qubit_idx, mapping)

        # Build 4x4 matrix for qu-quart
        # If pos=0 (lower level): gate acts on |0⟩,|1⟩, identity on |2⟩,|3⟩
        # If pos=1 (upper level): identity on |0⟩,|1⟩, gate acts on |2⟩,|3⟩
        if pos == 0:
            # Gate on lower two levels
            ququart_gate = qt.tensor(gate_qobj, qt.qeye(2))
        else:
            # Gate on upper two levels
            ququart_gate = qt.tensor(qt.qeye(2), gate_qobj)

        return (ququart_gate, [ququart_idx], False)

    elif len(qubit_indices) == 2:
        # Two-qubit gate
        q1, q2 = qubit_indices
        ququart1_idx, pos1 = get_ququart_index(q1, mapping)
        ququart2_idx, pos2 = get_ququart_index(q2, mapping)

        if ququart1_idx == ququart2_idx:
            # Both qubits in same qu-quart — becomes single qu-quart gate (4x4)
            if pos1 == 0 and pos2 == 1:
                # Natural ordering: gate qubit 0 → pos 0, gate qubit 1 → pos 1
                ququart_gate = gate_qobj
            elif pos1 == 1 and pos2 == 0:
                # Reversed: conjugate by SWAP to reorder qubit subspaces
                swap_qobj = qt.Qobj(SWAP_2Q)
                ququart_gate = swap_qobj * gate_qobj * swap_qobj
            else:
                raise ValueError(
                    f"Both qubits mapped to same position {pos1} in ququart {ququart1_idx}"
                )
            return (ququart_gate, [ququart1_idx], False)

        else:
            # Cross-qu-quart gate — embed into 16x16 two-ququart space.
            # Step 1: build in standard 4-qubit basis (index = 8*q0+4*q1+2*q2+q3)
            M_std = embed_cross_ququart_gate(gate_matrix, q1, q2).full()
            # Step 2: change basis to ququart-level basis (level = pos0_val + 2*pos1_val)
            M_qq = standard_to_ququart_basis(M_std, mapping)
            ququart_gate = qt.Qobj(M_qq, dims=[[4, 4], [4, 4]])
            sorted_indices = sorted([ququart1_idx, ququart2_idx])
            return (ququart_gate, sorted_indices, True)

    else:
        raise ValueError(f"Gates with {len(qubit_indices)} qubits not supported")

########### DISPLAY & CONVERSION ################
def qubit_to_ququart_circuit(qubit_circuit: QubitCircuit,
                             mapping: List[Tuple[int, int]] = None) -> QuQuartCircuit:
    """
    Main Conversion Function: Convert qubit circuit to qu-quart circuit.

    Parameters:
    -----------
    qubit_circuit : QubitCircuit
        The input qubit circuit
    mapping : List[Tuple[int, int]], optional
        The qubit-to-ququart mapping. If None, BMO will be used to find optimal mapping.

    Returns:
    --------
    QuQuartCircuit
        The converted qu-quart circuit with optimization metrics

    Algorithm:
    ----------
    1. If no mapping provided, use BMO to find optimal pairing
    2. For each layer in qubit circuit:
        - Convert single-qubit gates to qu-quart representation
        - Convert 2-qubit gates:
          * Same qu-quart → single qu-quart gate
          * Different qu-quarts → multi-qu-quart gate (flagged)
    3. Track cross-qu-quart gate count
    """
    # Step 1: Determine mapping
    if mapping is None:
        print("\nNo mapping was provided. We'll now run the Best Mapping Optimization...\n")
        mapping = best_mapping_optimization(qubit_circuit)
    else:
        print(f"Using provided mapping: {mapping}")

    # Validate mapping
    num_ququarts = len(mapping)
    mapped_qubits = set()
    for pair in mapping:
        mapped_qubits.update(pair)

    if len(mapped_qubits) != qubit_circuit.num_qubits:
        raise ValueError("Mapping does not cover all qubits")

    # Initialize qu-quart circuit
    ququart_circuit = QuQuartCircuit(num_ququarts, mapping)

    # Step 2: Convert each layer
    print(f"\nConverting {len(qubit_circuit.layers)} layers...")

    for layer_idx, layer in enumerate(qubit_circuit.layers):
        ququart_layer = []

        for gate_matrix, qubit_indices in layer:
            ququart_gate, ququart_indices, is_cross = qubit_gate_to_ququart(
                gate_matrix, qubit_indices, mapping
            )

            ququart_layer.append((ququart_gate.full(), ququart_indices))

            if is_cross:
                print(f"  Layer {layer_idx}: Cross-qu-quart gate on qubits {qubit_indices} "
                      f"-> qu-quarts {ququart_indices}")

        ququart_circuit.add_layer(ququart_layer)

    # Step 3: Report metrics
    print(f"\nConversion complete!")
    print(f"  Total layers: {len(ququart_circuit.layers)}")
    print(f"  Cross-qu-quart gates: {ququart_circuit.cross_ququart_gate_count}")

    return ququart_circuit

def display_ququart_circuit(qqc: QuQuartCircuit):
    """
    Display the ququart circuit with mapping and per-layer gate details.

    Shows:
    1. The qubit-to-ququart mapping with ququart level encoding
    2. For each layer, each gate's identified name, matrix, and target ququart(s)
    """
    print("\n" + "=" * 60)
    print("QuQuart Circuit Details")
    print("=" * 60)

    # 1. Display mapping
    # Level encoding: level = pos0_val + 2*pos1_val  (pos0 is LSB, pos1 is MSB)
    print("\nQubit-to-QuQuart Mapping (Optimal):\n")
    for qq_idx, (q1, q2) in enumerate(qqc.mapping):
        print(f"  QQ{qq_idx}: qubit {q1} (pos 0, LSB) + qubit {q2} (pos 1, MSB)")
        print(f"    level = pos0_val + 2*pos1_val")
        print(f"    |0>_QQ{qq_idx} = |0>_q{q1} |0>_q{q2}  (pos0=0, pos1=0 -> 0+0=0)")
        print(f"    |1>_QQ{qq_idx} = |1>_q{q1} |0>_q{q2}  (pos0=1, pos1=0 -> 1+0=1)")
        print(f"    |2>_QQ{qq_idx} = |0>_q{q1} |1>_q{q2}  (pos0=0, pos1=1 -> 0+2=2)")
        print(f"    |3>_QQ{qq_idx} = |1>_q{q1} |1>_q{q2}  (pos0=1, pos1=1 -> 1+2=3)")
        print()

    print(f"  Total ququarts: {qqc.num_ququarts}")
    print(f"  Total layers:   {len(qqc.layers)}")
    print(f"  Cross-ququart gates: {qqc.cross_ququart_gate_count}")

    # 2. Display each layer
    print("\n" + "-" * 60)
    print("Layer-by-Layer Gates:")
    print("-" * 60)

    for layer_idx, layer in enumerate(qqc.layers):
        print(f"\n  Layer {layer_idx}:\n")

        for gate_matrix, ququart_indices in layer:
            is_cross = len(ququart_indices) > 1
            gate_name = identify_gate(gate_matrix)
            dim = gate_matrix.shape[0]

            if is_cross:
                target_str = f"QQ{ququart_indices[0]} x QQ{ququart_indices[1]}"
                tag = "cross-ququart"
            else:
                target_str = f"QQ{ququart_indices[0]}"
                tag = "local"

            print(f"    {gate_name} on {target_str} ({tag}, {dim}x{dim}):")

            def _fmt(M):
                D = M.real if np.allclose(M.imag, 0, atol=1e-10) else M
                return np.array2string(D, precision=4, suppress_small=True, prefix="")

            if is_cross:
                # gate_matrix is stored in ququart basis; recover standard basis too
                P = build_basis_permutation(qqc.mapping)
                M_std = P.T @ gate_matrix @ P

                print(f"      [Ququart basis  |QQ_A, QQ_B⟩  (level = pos0 + 2·pos1)]:")
                for line in _fmt(gate_matrix).split('\n'):
                    print(f"        {line}")
                print()
                print(f"      [Standard basis  |q0 q1 q2 q3⟩  (index = 8q0+4q1+2q2+q3)]:")
                for line in _fmt(M_std).split('\n'):
                    print(f"        {line}")
            else:
                for line in _fmt(gate_matrix).split('\n'):
                    print(f"      {line}")

            print()

    print("\n" + "=" * 60)


def create_example_circuit() -> QubitCircuit:
    """
    Create an example 4-qubit circuit for testing.

    Returns:
    --------
    QubitCircuit
        A sample circuit with various gate types
    """
    circuit = QubitCircuit(num_qubits=4)

    # Define common quantum gates using QuTiP
    I = qt.qeye(2).full()
    X = qt.sigmax().full()
    # Hadamard gate (manual definition)
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)

    # CNOT gate (Control-NOT)

    # Layer 0: Hadamard on all qubits
    circuit.add_layer([
        (H, [0]),
        (H, [1]),
        (H, [2]),
        (H, [3])
    ])

    # Layer 1: CNOT gates
    # Many interactions between q0-q1 and q2-q3 (should pair them together)
    circuit.add_layer([
        (KNOWN_GATES["CNOT"], [0, 3]),
        (KNOWN_GATES["CNOT"], [1, 2])
    ])

    # Layer 2: Single cross-pair interaction
    circuit.add_layer([
        (KNOWN_GATES["CNOT"], [2, 3])
    ])

    return circuit


if __name__ == "__main__":
    print("="*60)
    print("Qubit-to-QuQuart Circuit Mapper Demo")
    print("="*60)

    # Create example circuit
    print("\n1. Creating example 4-qubit circuit...\n")
    qubit_circuit = create_example_circuit()
    print(f"   {qubit_circuit}")

    # Convert to qu-quart circuit (BMO will find optimal mapping)
    print("\n2. Converting to qu-quart circuit...")
    ququart_circuit = qubit_to_ququart_circuit(qubit_circuit)

    # Display detailed results
    print("\n3. Displaying qu-quart circuit details...")
    display_ququart_circuit(ququart_circuit)
