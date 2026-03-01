"""
Unit Tests for Qubit-to-QuQuart Circuit Mapper

Run with: python test_mapper.py
"""

import numpy as np
import qutip as qt
from qubit_to_ququart_mapper import (
    QubitCircuit,
    QuQuartCircuit,
    best_mapping_optimization,
    get_ququart_index,
    qubit_to_ququart_circuit,
    identify_gate,
    display_ququart_circuit,
    SWAP_2Q,
)


def test_qubit_circuit_creation():
    """Test QubitCircuit initialization and layer addition"""
    print("\nTest 1: QubitCircuit creation...")

    circuit = QubitCircuit(num_qubits=4)
    assert circuit.num_qubits == 4
    assert len(circuit.layers) == 0

    # Hadamard gate (manual definition)
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
    circuit.add_layer([(H, [0]), (H, [1])])
    assert len(circuit.layers) == 1

    print("  ✓ QubitCircuit creation works correctly")


def test_bmo_simple():
    """Test BMO with simple obvious pairing"""
    print("\nTest 2: BMO with simple pairing...")

    circuit = QubitCircuit(num_qubits=4)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # Create strong coupling between (0,1) and (2,3)
    for _ in range(10):
        circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])

    # Add single cross-pair interaction
    circuit.add_layer([(CNOT, [0, 2])])

    mapping = best_mapping_optimization(circuit)

    # The optimal mapping should pair (0,1) and (2,3)
    assert (0, 1) in mapping or (1, 0) in mapping
    assert (2, 3) in mapping or (3, 2) in mapping

    print("  ✓ BMO correctly identifies optimal pairing")


def test_bmo_all_mappings():
    """Test that BMO considers all three possible mappings"""
    print("\nTest 3: BMO considers all possible mappings...")

    circuit = QubitCircuit(num_qubits=4)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # Create circuit where (0,3) and (1,2) should be paired
    for _ in range(5):
        circuit.add_layer([(CNOT, [0, 3]), (CNOT, [1, 2])])

    mapping = best_mapping_optimization(circuit)

    # Verify one of the expected pairings
    valid_mappings = [
        [(0, 3), (1, 2)],
        [(3, 0), (1, 2)],
        [(0, 3), (2, 1)],
        [(3, 0), (2, 1)],
        [(1, 2), (0, 3)],
        [(2, 1), (0, 3)],
        [(1, 2), (3, 0)],
        [(2, 1), (3, 0)]
    ]

    assert mapping in valid_mappings or \
           [mapping[1], mapping[0]] in valid_mappings

    print("  ✓ BMO correctly handles different pairing scenarios")


def test_get_ququart_index():
    """Test qubit to qu-quart index mapping"""
    print("\nTest 4: Qubit to qu-quart index mapping...")

    mapping = [(0, 1), (2, 3)]

    assert get_ququart_index(0, mapping) == (0, 0)
    assert get_ququart_index(1, mapping) == (0, 1)
    assert get_ququart_index(2, mapping) == (1, 0)
    assert get_ququart_index(3, mapping) == (1, 1)

    print("  ✓ Qubit index mapping works correctly")


def test_single_qubit_conversion():
    """Test conversion of single-qubit gates"""
    print("\nTest 5: Single-qubit gate conversion...")

    circuit = QubitCircuit(num_qubits=4)
    X = qt.sigmax().full()

    circuit.add_layer([(X, [0]), (X, [1]), (X, [2]), (X, [3])])

    mapping = [(0, 1), (2, 3)]
    ququart = qubit_to_ququart_circuit(circuit, mapping=mapping)

    # Single-qubit gates should not create cross-qu-quart gates
    assert ququart.cross_ququart_gate_count == 0
    assert len(ququart.layers) == 1

    print("  ✓ Single-qubit gates convert correctly")


def test_within_ququart_cnot():
    """Test CNOT gates within same qu-quart"""
    print("\nTest 6: Within-qu-quart CNOT gates...")

    circuit = QubitCircuit(num_qubits=4)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # Gates within same qu-quart pairs
    circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])
    circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])

    mapping = [(0, 1), (2, 3)]
    ququart = qubit_to_ququart_circuit(circuit, mapping=mapping)

    # Within-qu-quart gates should not be cross-qu-quart
    assert ququart.cross_ququart_gate_count == 0

    print("  ✓ Within-qu-quart CNOTs identified correctly")


def test_cross_ququart_cnot():
    """Test CNOT gates across different qu-quarts"""
    print("\nTest 7: Cross-qu-quart CNOT gates...")

    circuit = QubitCircuit(num_qubits=4)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # Gates across qu-quart boundaries
    circuit.add_layer([(CNOT, [0, 2]), (CNOT, [1, 3])])
    circuit.add_layer([(CNOT, [0, 3])])

    mapping = [(0, 1), (2, 3)]
    ququart = qubit_to_ququart_circuit(circuit, mapping=mapping)

    # Should have 3 cross-qu-quart gates
    assert ququart.cross_ququart_gate_count == 3

    print("  ✓ Cross-qu-quart CNOTs identified correctly")


def test_mixed_layer():
    """Test layer with both within and cross qu-quart gates"""
    print("\nTest 8: Mixed layers...")

    circuit = QubitCircuit(num_qubits=4)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # Layer with both types
    circuit.add_layer([(CNOT, [0, 1])])  # Within qu-quart
    circuit.add_layer([(CNOT, [0, 2])])  # Cross qu-quart

    mapping = [(0, 1), (2, 3)]
    ququart = qubit_to_ququart_circuit(circuit, mapping=mapping)

    assert ququart.cross_ququart_gate_count == 1

    print("  ✓ Mixed layers handled correctly")


def test_optimization_benefit():
    """Test that optimization actually reduces cross-qu-quart gates"""
    print("\nTest 9: Optimization benefit...")

    circuit = QubitCircuit(num_qubits=4)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # Create circuit with obvious best pairing
    for _ in range(10):
        circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])
    circuit.add_layer([(CNOT, [0, 2])])

    # Test with optimal mapping
    optimal_mapping = [(0, 1), (2, 3)]
    ququart_optimal = qubit_to_ququart_circuit(circuit, mapping=optimal_mapping)

    # Test with suboptimal mapping
    suboptimal_mapping = [(0, 2), (1, 3)]
    ququart_suboptimal = qubit_to_ququart_circuit(circuit, mapping=suboptimal_mapping)

    # Optimal should have fewer cross-qu-quart gates
    assert ququart_optimal.cross_ququart_gate_count < ququart_suboptimal.cross_ququart_gate_count

    print(f"  Optimal: {ququart_optimal.cross_ququart_gate_count} cross-gates")
    print(f"  Suboptimal: {ququart_suboptimal.cross_ququart_gate_count} cross-gates")
    print("  ✓ Optimization provides measurable benefit")


def test_no_two_qubit_gates():
    """Test circuit with only single-qubit gates"""
    print("\nTest 10: Circuit with no 2-qubit gates...")

    circuit = QubitCircuit(num_qubits=4)
    # Hadamard gate (manual definition)
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
    X = qt.sigmax().full()

    circuit.add_layer([(H, [i]) for i in range(4)])
    circuit.add_layer([(X, [i]) for i in range(4)])

    ququart = qubit_to_ququart_circuit(circuit)

    # Should have no cross-qu-quart gates
    assert ququart.cross_ququart_gate_count == 0

    print("  ✓ Circuits with only single-qubit gates handled correctly")


def test_within_ququart_reversed_ordering():
    """Test CNOT with reversed qubit ordering within same ququart"""
    print("\nTest 11: Within-ququart reversed qubit ordering...")

    circuit = QubitCircuit(num_qubits=4)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    # CNOT[1, 0]: qubit 1 is control (pos=1), qubit 0 is target (pos=0)
    # With mapping [(0,1), (2,3)], needs SWAP reordering
    circuit.add_layer([(CNOT, [1, 0])])

    mapping = [(0, 1), (2, 3)]
    ququart = qubit_to_ququart_circuit(circuit, mapping=mapping)

    assert ququart.cross_ququart_gate_count == 0

    # The gate should be SWAP @ CNOT @ SWAP (reversed CNOT)
    gate_matrix = ququart.layers[0][0][0]
    expected = SWAP_2Q @ CNOT @ SWAP_2Q
    assert np.allclose(gate_matrix, expected), \
        f"Reversed CNOT matrix mismatch:\n{gate_matrix}\nvs expected:\n{expected}"

    print("  ✓ Reversed qubit ordering handled correctly")


def test_cross_ququart_embedding_dimensions():
    """Test that cross-ququart gates produce 16x16 unitary matrices"""
    print("\nTest 12: Cross-ququart gate embedding dimensions...")

    circuit = QubitCircuit(num_qubits=4)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    circuit.add_layer([(CNOT, [0, 2])])  # Cross-ququart

    mapping = [(0, 1), (2, 3)]
    ququart = qubit_to_ququart_circuit(circuit, mapping=mapping)

    gate_matrix = ququart.layers[0][0][0]
    assert gate_matrix.shape == (16, 16), \
        f"Expected 16x16, got {gate_matrix.shape}"

    # Verify unitarity
    product = gate_matrix @ gate_matrix.conj().T
    assert np.allclose(product, np.eye(16)), "Embedded gate is not unitary"

    print("  ✓ Cross-ququart embedding produces correct 16x16 unitary")


def test_cross_ququart_all_position_variants():
    """Test cross-ququart embedding for all 4 position combinations"""
    print("\nTest 13: Cross-ququart all position variants...")

    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    mapping = [(0, 1), (2, 3)]
    # (pos1, pos2) variants: (0,0), (0,1), (1,0), (1,1)
    cross_pairs = [
        ([0, 2], "pos(0,0)"),  # q0=pos0 in QQ0, q2=pos0 in QQ1
        ([0, 3], "pos(0,1)"),  # q0=pos0 in QQ0, q3=pos1 in QQ1
        ([1, 2], "pos(1,0)"),  # q1=pos1 in QQ0, q2=pos0 in QQ1
        ([1, 3], "pos(1,1)"),  # q1=pos1 in QQ0, q3=pos1 in QQ1
    ]

    for qubit_indices, label in cross_pairs:
        circuit = QubitCircuit(num_qubits=4)
        circuit.add_layer([(CNOT, qubit_indices)])
        ququart = qubit_to_ququart_circuit(circuit, mapping=mapping)

        gate_matrix = ququart.layers[0][0][0]
        assert gate_matrix.shape == (16, 16), \
            f"{label}: Expected 16x16, got {gate_matrix.shape}"

        product = gate_matrix @ gate_matrix.conj().T
        assert np.allclose(product, np.eye(16)), \
            f"{label}: Embedded gate is not unitary"

    print("  ✓ All 4 cross-ququart position variants produce valid unitaries")


def test_identify_gate():
    """Test gate identification from known gate dictionary"""
    print("\nTest 14: Gate identification...")

    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]
    ], dtype=complex)

    assert identify_gate(H) == "H"
    assert identify_gate(CNOT) == "CNOT"
    assert identify_gate(np.eye(2, dtype=complex)) == "I"

    # Unknown gate should return U(NxN)
    random_gate = np.array([[0.5, 0.5], [0.5, -0.5]], dtype=complex)
    result = identify_gate(random_gate)
    assert result.startswith("U("), f"Expected U(NxN), got {result}"

    print("  ✓ Gate identification works correctly")


def run_all_tests():
    """Run all tests"""
    print("="*70)
    print(" RUNNING UNIT TESTS FOR QUBIT-TO-QUQUART MAPPER ")
    print("="*70)

    tests = [
        test_qubit_circuit_creation,
        test_bmo_simple,
        test_bmo_all_mappings,
        test_get_ququart_index,
        test_single_qubit_conversion,
        test_within_ququart_cnot,
        test_cross_ququart_cnot,
        test_mixed_layer,
        test_optimization_benefit,
        test_no_two_qubit_gates,
        test_within_ququart_reversed_ordering,
        test_cross_ququart_embedding_dimensions,
        test_cross_ququart_all_position_variants,
        test_identify_gate,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Test error: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f" RESULTS: {passed} passed, {failed} failed ")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)