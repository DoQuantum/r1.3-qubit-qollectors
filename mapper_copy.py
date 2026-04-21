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
    def __init__(self, num_ququarts: int, mapping: List[Tuple[int, int]],
                 num_original_qubits: int = None):
        self.num_ququarts = num_ququarts
        self.mapping = mapping  # Which qubits are paired into each qu-quart
        self.num_original_qubits = (num_original_qubits if num_original_qubits is not None
                                    else 2 * num_ququarts)
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
        has_ancilla = self.num_original_qubits < 2 * self.num_ququarts
        ancilla_note = " (1 ancilla)" if has_ancilla else ""
        return (f"QuQuartCircuit({self.num_ququarts} qu-quarts from "
                f"{self.num_original_qubits} qubits{ancilla_note}, "
                f"{len(self.layers)} layers, "
                f"{self.cross_ququart_gate_count} cross-qu-quart gates)")


# --- BMO helper functions ---

def _all_perfect_matchings(nodes: list):
    """Recursively generate all perfect matchings of a list of nodes."""
    if not nodes:
        yield []
        return
    first = nodes[0]
    rest = nodes[1:]
    for i, second in enumerate(rest):
        remaining = rest[:i] + rest[i + 1:]
        for sub in _all_perfect_matchings(remaining):
            yield [(first, second)] + sub


def _greedy_matching(nodes: list, interaction_counts: dict) -> list:
    """
    Greedy maximum-weight perfect matching.
    Iteratively picks the highest-interaction unmatched pair, then pairs
    any remaining unmatched nodes arbitrarily.
    """
    sorted_pairs = sorted(interaction_counts.keys(), key=lambda p: -interaction_counts[p])
    unmatched = set(nodes)
    matching = []
    for (i, j) in sorted_pairs:
        if i in unmatched and j in unmatched:
            matching.append((i, j))
            unmatched.discard(i)
            unmatched.discard(j)
    unmatched_list = sorted(unmatched)
    while len(unmatched_list) >= 2:
        matching.append((unmatched_list[0], unmatched_list[1]))
        unmatched_list = unmatched_list[2:]
    return matching


def best_mapping_optimization(qubit_circuit: QubitCircuit) -> List[Tuple[int, int]]:
    """
    Best Mapping Optimization (BMO) Function

    Finds the optimal qubit pairing to minimize cross-qu-quart gates by analyzing
    the frequency of 2-qubit gates between different qubit pairs.

    Supports any number of qubits (even or odd). For an odd number of qubits,
    one ancilla qubit (index = num_qubits) is automatically added so that all
    qubits form complete ququart pairs.

    Parameters:
    -----------
    qubit_circuit : QubitCircuit
        The input qubit circuit to analyze

    Returns:
    --------
    List[Tuple[int, int]]
        Optimal pairing of qubits. For odd qubit counts, one pair will contain
        the ancilla qubit at index qubit_circuit.num_qubits.

    Algorithm:
    ----------
    1. If odd qubit count, add a virtual ancilla qubit with no interactions.
    2. Build weighted interaction graph (edge weight = 2-qubit gate count).
    3. For n <= 12 qubits: enumerate all perfect matchings and pick the minimum
       cross-pair cost. For n > 12: use greedy maximum-weight matching.
    """
    num_qubits = qubit_circuit.num_qubits
    has_ancilla = (num_qubits % 2 == 1)
    effective_n = num_qubits + 1 if has_ancilla else num_qubits

    # Build interaction graph (ancilla qubit has zero interactions)
    interaction_counts = {}
    for i in range(effective_n):
        for j in range(i + 1, effective_n):
            interaction_counts[(i, j)] = 0

    for layer in qubit_circuit.layers:
        for gate_matrix, qubit_indices in layer:
            if len(qubit_indices) == 2:
                q1, q2 = sorted(qubit_indices)
                if (q1, q2) in interaction_counts:
                    interaction_counts[(q1, q2)] += 1

    nodes = list(range(effective_n))

    # Enumerate all perfect matchings for small n; greedy for larger n.
    # (n-1)!! matchings: n=12 → 10395, n=14 → 135135
    _ENUM_THRESHOLD = 12

    if effective_n <= _ENUM_THRESHOLD:
        best_pairing = None
        min_cross = float('inf')
        for matching in _all_perfect_matchings(nodes):
            pair_map = {}
            for qq_idx, (a, b) in enumerate(matching):
                pair_map[a] = qq_idx
                pair_map[b] = qq_idx
            cross = sum(
                cnt for (q1, q2), cnt in interaction_counts.items()
                if pair_map[q1] != pair_map[q2]
            )
            if cross < min_cross:
                min_cross = cross
                best_pairing = matching
    else:
        best_pairing = _greedy_matching(nodes, interaction_counts)
        pair_map = {}
        for qq_idx, (a, b) in enumerate(best_pairing):
            pair_map[a] = qq_idx
            pair_map[b] = qq_idx
        min_cross = sum(
            cnt for (q1, q2), cnt in interaction_counts.items()
            if pair_map[q1] != pair_map[q2]
        )

    nonzero_interactions = {k: v for k, v in interaction_counts.items() if v > 0}

    if has_ancilla:
        print(f"BMO Analysis (odd qubit count — ancilla qubit {num_qubits} added):")
    else:
        print(f"BMO Analysis:")
    print(f"  Interaction counts: {nonzero_interactions}\n")
    print(f"  Most optimal pairing: {best_pairing}")
    print(f"  Cross-qu-quart gates: {min_cross}")

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
    Build the permutation matrix P that converts from the standard computational
    basis to the ququart-level basis, for any number of ququart pairs.

    Standard basis index:  s = sum_k  qubit_k_val * 2^(n_qubits - 1 - k)
                              (qubit 0 has the highest place value)
    Ququart-level index:   qq = sum_i  QQ_i * 4^(n_ququarts - 1 - i)
                               where QQ_i = pos0_val + 2 * pos1_val

    Usage:
        P = build_basis_permutation(mapping)
        M_ququart = P @ M_standard @ P.T     (P.T == P^{-1} since P is a permutation)
        M_standard = P.T @ M_ququart @ P
    """
    n_ququarts = len(mapping)
    n_qubits = 2 * n_ququarts
    dim = 4 ** n_ququarts

    P = np.zeros((dim, dim), dtype=complex)
    for s in range(dim):
        q = [(s >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
        qq_idx = 0
        for (a, b) in mapping:
            QQ = q[a] + 2 * q[b]        # level = pos0_val + 2*pos1_val
            qq_idx = qq_idx * 4 + QQ
        P[qq_idx, s] = 1.0
    return P


def standard_to_ququart_basis(M_std: np.ndarray,
                               mapping: List[Tuple[int, int]]) -> np.ndarray:
    """
    Convert a matrix from the standard computational basis to the ququart-level basis.

    Parameters:
    -----------
    M_std : np.ndarray
        Square matrix in the standard |q0 q1 ... q_{n-1}⟩ basis
    mapping : List[Tuple[int, int]]
        The qubit-to-ququart mapping (determines basis permutation)

    Returns:
    --------
    np.ndarray
        Matrix in the ququart-level |QQ_0, QQ_1, ...⟩ basis
    """
    P = build_basis_permutation(mapping)
    return P @ M_std @ P.T


#Instead of constructing the 16x16 matrix from scratch, use tensoring and swaps within ququarts
#
def embed_cross_ququart_gate(gate_matrix: np.ndarray, local_a: int, local_b: int) -> qt.Qobj:
    """
    Embed a 2-qubit gate into the 16-dimensional local space of two ququarts.

    The local 4-qubit basis is:
        local qubit 0 = pos-0 of QQ_A  (first ququart)
        local qubit 1 = pos-1 of QQ_A
        local qubit 2 = pos-0 of QQ_B  (second ququart)
        local qubit 3 = pos-1 of QQ_B

    Local standard basis index = 8*q_loc0 + 4*q_loc1 + 2*q_loc2 + q_loc3

    The gate acts on local qubit local_a (gate's first/control qubit, higher place value
    in the gate index) and local qubit local_b (gate's second/target qubit, lower place value).
    The remaining two local qubits are spectators (identity).

    Parameters:
    -----------
    gate_matrix : np.ndarray
        4x4 unitary gate matrix
    local_a : int
        Local qubit index (0–3) for the gate's first qubit
    local_b : int
        Local qubit index (0–3) for the gate's second qubit

    Returns:
    --------
    qt.Qobj
        16x16 matrix in the local standard basis, dims=[[4,4],[4,4]]
    """
    G = np.array(gate_matrix, dtype=complex)
    M = np.zeros((16, 16), dtype=complex)

    for i_out in range(16):
        for i_in in range(16):
            # Extract 4-bit representation: bits[k] = bit value of local qubit k
            bits_out = [(i_out >> (3 - k)) & 1 for k in range(4)]
            bits_in  = [(i_in  >> (3 - k)) & 1 for k in range(4)]

            # Spectator local qubits must have matching bits
            spectators_match = all(
                bits_out[k] == bits_in[k]
                for k in range(4)
                if k != local_a and k != local_b
            )
            if not spectators_match:
                continue

            # Map active qubit bits to the gate's row/column indices
            # local_a is the gate's first qubit (×2 place value), local_b is second (×1 place value)
            g_row = bits_out[local_a] * 2 + bits_out[local_b]
            g_col = bits_in[local_a]  * 2 + bits_in[local_b]
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
            # Cross-qu-quart gate — embed into the 16-dim two-ququart local space.
            #
            # Local qubit ordering:
            #   local 0 = pos-0 of QQ_A (lower-indexed ququart)
            #   local 1 = pos-1 of QQ_A
            #   local 2 = pos-0 of QQ_B (higher-indexed ququart)
            #   local 3 = pos-1 of QQ_B
            #
            # The gate's first qubit (q1) contributes the ×2 place value; second qubit (q2) contributes ×1.
            sorted_indices = sorted([ququart1_idx, ququart2_idx])
            _LOCAL_MAPPING = [(0, 1), (2, 3)]

            if ququart1_idx <= ququart2_idx:
                # q1 is in QQ_A (local 0–1), q2 is in QQ_B (local 2–3)
                local_a = pos1        # q1 at local position pos1 (0 or 1)
                local_b = 2 + pos2    # q2 at local position 2+pos2 (2 or 3)
            else:
                # q1 is in QQ_B (local 2–3), q2 is in QQ_A (local 0–1)
                local_a = 2 + pos1    # q1 at local position 2+pos1 (2 or 3)
                local_b = pos2         # q2 at local position pos2 (0 or 1)

            M_std = embed_cross_ququart_gate(gate_matrix, local_a, local_b).full()
            M_qq = standard_to_ququart_basis(M_std, _LOCAL_MAPPING)
            ququart_gate = qt.Qobj(M_qq, dims=[[4, 4], [4, 4]])
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
        The input qubit circuit (any number of qubits, even or odd)
    mapping : List[Tuple[int, int]], optional
        The qubit-to-ququart mapping. If None, BMO will find the optimal mapping.
        For odd qubit counts, the mapping must include an ancilla qubit at index
        num_qubits (or BMO will add one automatically).

    Returns:
    --------
    QuQuartCircuit
        The converted qu-quart circuit with optimization metrics

    Algorithm:
    ----------
    1. If no mapping provided, use BMO to find optimal pairing
       (BMO adds an ancilla for odd qubit counts automatically).
    2. For each layer in qubit circuit:
        - Convert single-qubit gates to qu-quart representation
        - Convert 2-qubit gates:
          * Same qu-quart → single qu-quart gate
          * Different qu-quarts → 16x16 multi-qu-quart gate (flagged)
    3. Track cross-qu-quart gate count
    """
    # Step 1: Determine mapping
    if mapping is None:
        print("\nNo mapping was provided. We'll now run the Best Mapping Optimization...\n")
        mapping = best_mapping_optimization(qubit_circuit)
    else:
        print(f"Using provided mapping: {mapping}")

    # Validate mapping covers all original qubits
    original_qubits = set(range(qubit_circuit.num_qubits))
    mapped_qubits = set()
    for pair in mapping:
        mapped_qubits.update(pair)

    if not original_qubits.issubset(mapped_qubits):
        raise ValueError("Mapping does not cover all qubits")

    num_ququarts = len(mapping)

    # Initialize qu-quart circuit
    ququart_circuit = QuQuartCircuit(num_ququarts, mapping,
                                     num_original_qubits=qubit_circuit.num_qubits)

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
    # Level encoding: level = pos0_val + 2*pos1_val
    ancilla_qubits = set(range(qqc.num_original_qubits, 2 * qqc.num_ququarts))

    print("\nQubit-to-QuQuart Mapping (Optimal):\n")
    for qq_idx, (q1, q2) in enumerate(qqc.mapping):
        q1_note = " [ancilla]" if q1 in ancilla_qubits else ""
        q2_note = " [ancilla]" if q2 in ancilla_qubits else ""
        print(f"  QQ{qq_idx}: qubit {q1} (pos 0){q1_note} + qubit {q2} (pos 1){q2_note}")
        print(f"    level = pos0_val + 2*pos1_val")
        print(f"    |0>_QQ{qq_idx} = |0>_q{q1} |0>_q{q2}  (pos0=0, pos1=0 -> 0+0=0)")
        print(f"    |1>_QQ{qq_idx} = |1>_q{q1} |0>_q{q2}  (pos0=1, pos1=0 -> 1+0=1)")
        print(f"    |2>_QQ{qq_idx} = |0>_q{q1} |1>_q{q2}  (pos0=0, pos1=1 -> 0+2=2)")
        print(f"    |3>_QQ{qq_idx} = |1>_q{q1} |1>_q{q2}  (pos0=1, pos1=1 -> 1+2=3)")
        print()

    print(f"  Total ququarts: {qqc.num_ququarts}")
    print(f"  Total layers:   {len(qqc.layers)}")
    print(f"  Cross-ququart gates: {qqc.cross_ququart_gate_count}")
    if ancilla_qubits:
        print(f"  Ancilla qubit(s): {sorted(ancilla_qubits)}")

    # 2. Display each layer
    print("\n" + "-" * 60)
    print("Layer-by-Layer Gates:")
    print("-" * 60)

    # Local 16x16 permutation for recovering standard basis of cross-ququart gates
    _LOCAL_MAPPING = [(0, 1), (2, 3)]
    P_local = build_basis_permutation(_LOCAL_MAPPING)

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
                # gate_matrix is in the local ququart basis; recover local standard basis
                M_std_local = P_local.T @ gate_matrix @ P_local

                print(f"      [Ququart basis  |QQ_A, QQ_B⟩  (level = pos0 + 2·pos1)]:")
                for line in _fmt(gate_matrix).split('\n'):
                    print(f"        {line}")
                print()
                print(f"      [Local standard basis  |pos0_A, pos1_A, pos0_B, pos1_B⟩]:")
                for line in _fmt(M_std_local).split('\n'):
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
 