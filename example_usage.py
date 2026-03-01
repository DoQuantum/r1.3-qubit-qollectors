"""
Example Usage of Qubit-to-QuQuart Circuit Mapper

This script demonstrates various use cases of the qubit-to-ququart mapper.
"""

import numpy as np
import qutip as qt
from qubit_to_ququart_mapper import (
    QubitCircuit,
    best_mapping_optimization,
    qubit_to_ququart_circuit,
    create_example_circuit,
    display_ququart_circuit,
)


def example_1_basic_usage():
    """Example 1: Basic usage with automatic mapping optimization"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage with Automatic Mapping Optimization")
    print("="*70)

    # Create a simple circuit
    circuit = QubitCircuit(num_qubits=4)

    # Define gates
    # Hadamard gate (manual definition)
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # Build circuit with strong coupling between (q0,q1) and (q2,q3)
    circuit.add_layer([(H, [i]) for i in range(4)])
    circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])
    circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])
    circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])

    # Convert (mapping will be automatically optimized)
    ququart_circuit = qubit_to_ququart_circuit(circuit)

    # Display detailed gate-by-gate results
    display_ququart_circuit(ququart_circuit)

    print(f"\nResult: {ququart_circuit}")
    print(f"Expected mapping: [(0,1), (2,3)] due to high interaction within pairs")

def example_X_basic_usage():
    """Example 1: Basic usage with automatic mapping optimization"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage with Automatic Mapping Optimization")
    print("="*70)

    # Create a simple circuit
    circuit = QubitCircuit(num_qubits=4) 

    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    circuit.add_layer([(H, [0])])
    circuit.add_layer([(CNOT, [0, 2])])
    circuit.add_layer([(CNOT, [1, 3])])
    circuit.add_layer([(CNOT, [1, 2])])

def example_2_manual_mapping():
    """Example 2: Using a manually specified mapping"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Manual Mapping Specification")
    print("="*70)

    circuit = create_example_circuit()

    # Specify a custom mapping (may not be optimal)
    custom_mapping = [(0, 2), (1, 3)]  # Pair q0 with q2, q1 with q3

    print(f"\nUsing custom mapping: {custom_mapping}")
    ququart_circuit = qubit_to_ququart_circuit(circuit, mapping=custom_mapping)

    print(f"\nResult: {ququart_circuit}")
    print(f"Note: This mapping may result in more cross-qu-quart gates")


def example_3_different_topologies():
    """Example 3: Circuits with different interaction topologies"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Different Interaction Topologies")
    print("="*70)

    # Define gates
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    X = qt.sigmax().full()

    # Topology A: Linear chain (0-1-2-3)
    print("\nTopology A: Linear Chain (q0-q1-q2-q3)")
    circuit_a = QubitCircuit(num_qubits=4)
    circuit_a.add_layer([(CNOT, [0, 1])])
    circuit_a.add_layer([(CNOT, [1, 2])])
    circuit_a.add_layer([(CNOT, [2, 3])])

    ququart_a = qubit_to_ququart_circuit(circuit_a)
    print(f"Result: {ququart_a}")

    # Topology B: Star (all connect to q0)
    print("\n\nTopology B: Star (all qubits connect to q0)")
    circuit_b = QubitCircuit(num_qubits=4)
    circuit_b.add_layer([(CNOT, [0, 1])])
    circuit_b.add_layer([(CNOT, [0, 2])])
    circuit_b.add_layer([(CNOT, [0, 3])])

    ququart_b = qubit_to_ququart_circuit(circuit_b)
    print(f"Result: {ququart_b}")

    # Topology C: Complete graph (all pairs interact equally)
    print("\n\nTopology C: Complete Graph (all pairs interact)")
    circuit_c = QubitCircuit(num_qubits=4)
    circuit_c.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])
    circuit_c.add_layer([(CNOT, [0, 2]), (CNOT, [1, 3])])
    circuit_c.add_layer([(CNOT, [0, 3]), (CNOT, [1, 2])])

    ququart_c = qubit_to_ququart_circuit(circuit_c)
    print(f"Result: {ququart_c}")


def example_4_analyzing_optimization():
    """Example 4: Analyzing the optimization benefit"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Analyzing Optimization Benefit")
    print("="*70)

    # Create a circuit with clear optimal pairing
    circuit = QubitCircuit(num_qubits=4)

    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # Strong interaction between (0,1) and (2,3)
    for _ in range(5):
        circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])

    # Few cross-pair interactions
    circuit.add_layer([(CNOT, [0, 2])])

    print("\nCircuit has:")
    print("  - 10 gates within pairs (0,1) and (2,3)")
    print("  - 1 gate crossing pairs")

    # Test all three possible mappings
    mappings = [
        [(0, 1), (2, 3)],  # Optimal
        [(0, 2), (1, 3)],  # Suboptimal
        [(0, 3), (1, 2)]   # Suboptimal
    ]

    results = []
    for mapping in mappings:
        print(f"\n--- Testing mapping {mapping} ---")
        ququart = qubit_to_ququart_circuit(circuit, mapping=mapping)
        results.append((mapping, ququart.cross_ququart_gate_count))

    print("\n" + "-"*70)
    print("Comparison of Mappings:")
    print("-"*70)
    for mapping, cross_gates in results:
        print(f"  {mapping}: {cross_gates} cross-qu-quart gates")

    best = min(results, key=lambda x: x[1])
    print(f"\nBest mapping: {best[0]} with {best[1]} cross-qu-quart gates")


def example_5_edge_cases():
    """Example 5: Edge cases and special scenarios"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Edge Cases and Special Scenarios")
    print("="*70)

    # Case A: No 2-qubit gates (only single-qubit gates)
    print("\nCase A: Circuit with only single-qubit gates")
    circuit_a = QubitCircuit(num_qubits=4)
    X = qt.sigmax().full()
    # Hadamard gate (manual definition)
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)

    circuit_a.add_layer([(H, [i]) for i in range(4)])
    circuit_a.add_layer([(X, [i]) for i in range(4)])

    ququart_a = qubit_to_ququart_circuit(circuit_a)
    print(f"Result: {ququart_a}")
    print(f"Note: With no 2-qubit gates, any mapping is equivalent")

    # Case B: Isolated qubit (one qubit never interacts)
    print("\n\nCase B: Circuit with isolated qubit (q3 never used)")
    circuit_b = QubitCircuit(num_qubits=4)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    circuit_b.add_layer([(CNOT, [0, 1]), (CNOT, [0, 1])])
    circuit_b.add_layer([(CNOT, [1, 2])])
    circuit_b.add_layer([(H, [3])])  # q3 only has single-qubit gate

    ququart_b = qubit_to_ququart_circuit(circuit_b)
    print(f"Result: {ququart_b}")
    print(f"Note: q3 will be paired with another qubit despite no 2-qubit interactions")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" QUBIT-TO-QUQUART CIRCUIT MAPPER: COMPREHENSIVE EXAMPLES ")
    print("="*70)

    example_1_basic_usage()
    example_2_manual_mapping()
    example_3_different_topologies()
    example_4_analyzing_optimization()
    example_5_edge_cases()

    print("\n" + "="*70)
    print(" All examples completed successfully! ")
    print("="*70)