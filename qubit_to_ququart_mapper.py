"""
Qubit-to-QuQuart Circuit Mapper

This module converts qubit circuits (2-level systems) into qu-quart circuits
(4-level systems) while minimizing cross-qu-quart gate operations using QuTiP.

Author: Quantum Research Team
Date: 2026-01-22
"""

import numpy as np
import qutip as qt
from typing import List, Tuple, Dict
from itertools import combinations


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

    def add_layer(self, gates: List[Tuple[np.ndarray, List[int]]], is_cross_ququart: bool = False):
        """
        Add a layer of gates to the qu-quart circuit.

        Parameters:
        -----------
        gates : List[Tuple[np.ndarray, List[int]]]
            List of (gate_matrix, [ququart_indices]) tuples
        is_cross_ququart : bool
            Whether this layer contains cross-qu-quart gates
        """
        self.layers.append(gates)
        if is_cross_ququart:
            self.cross_ququart_gate_count += len(gates)

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
    print(f"  Interaction counts: {interaction_counts}")
    print(f"  Best pairing: {best_pairing}")
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
            # Both qubits in same qu-quart - becomes single qu-quart gate
            # The gate already acts on a 4x4 space (2 qubits = 4 levels)
            ququart_gate = gate_qobj
            return (ququart_gate, [ququart1_idx], False)
        else:
            # Qubits in different qu-quarts - cross-qu-quart gate
            # Need to properly embed the 2-qubit gate into 2 qu-quarts (4x4 each)

            # Determine the ordering
            if ququart1_idx < ququart2_idx:
                # Expand each qubit to qu-quart level
                if pos1 == 0:
                    gate1 = qt.tensor(qt.qeye(2), qt.qeye(2))  # Placeholder
                else:
                    gate1 = qt.tensor(qt.qeye(2), qt.qeye(2))

                if pos2 == 0:
                    gate2 = qt.tensor(qt.qeye(2), qt.qeye(2))
                else:
                    gate2 = qt.tensor(qt.qeye(2), qt.qeye(2))

                # For cross-qu-quart gates, we need to embed properly
                # This is complex - for now, return the gate as-is with both indices
                ququart_gate = gate_qobj
                return (ququart_gate, [ququart1_idx, ququart2_idx], True)
            else:
                ququart_gate = gate_qobj
                return (ququart_gate, [ququart2_idx, ququart1_idx], True)

    else:
        raise ValueError(f"Gates with {len(qubit_indices)} qubits not supported")


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
        print("No mapping provided. Running Best Mapping Optimization...")
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
        layer_has_cross_gate = False

        for gate_matrix, qubit_indices in layer:
            ququart_gate, ququart_indices, is_cross = qubit_gate_to_ququart(
                gate_matrix, qubit_indices, mapping
            )

            ququart_layer.append((ququart_gate.full(), ququart_indices))

            if is_cross:
                layer_has_cross_gate = True
                print(f"  Layer {layer_idx}: Cross-qu-quart gate on qubits {qubit_indices} "
                      f"→ qu-quarts {ququart_indices}")

        ququart_circuit.add_layer(ququart_layer, is_cross_ququart=layer_has_cross_gate)

    # Step 3: Report metrics
    print(f"\nConversion complete!")
    print(f"  Total layers: {len(ququart_circuit.layers)}")
    print(f"  Cross-qu-quart gates: {ququart_circuit.cross_ququart_gate_count}")

    return ququart_circuit


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
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

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
        (CNOT, [0, 1]),
        (CNOT, [2, 3])
    ])

    # Layer 2: More CNOT gates favoring same pairs
    circuit.add_layer([
        (CNOT, [0, 1]),
        (CNOT, [2, 3])
    ])

    # Layer 3: Single cross-pair interaction
    circuit.add_layer([
        (CNOT, [1, 2]),
        (I, [0]),
        (I, [3])
    ])

    # Layer 4: X gates
    circuit.add_layer([
        (X, [0]),
        (X, [2])
    ])

    return circuit


if __name__ == "__main__":
    print("="*60)
    print("Qubit-to-QuQuart Circuit Mapper Demo")
    print("="*60)

    # Create example circuit
    print("\n1. Creating example 4-qubit circuit...")
    qubit_circuit = create_example_circuit()
    print(f"   {qubit_circuit}")

    # Convert to qu-quart circuit (BMO will find optimal mapping)
    print("\n2. Converting to qu-quart circuit...")
    ququart_circuit = qubit_to_ququart_circuit(qubit_circuit)

    # Display results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Input:  {qubit_circuit}")
    print(f"Output: {ququart_circuit}")
    print(f"\nOptimization achieved:")
    print(f"  Mapping: {ququart_circuit.mapping}")
    print(f"  Cross-qu-quart gates: {ququart_circuit.cross_ququart_gate_count}")
    print("="*60)
