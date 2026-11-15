import unittest
import warnings

from qiskit import QuantumCircuit

from parser import build_interaction_graph, parse_qasm, parse_qiskit


class InteractionGraphTests(unittest.TestCase):
    def test_build_interaction_graph_weights_edges_and_keeps_isolated_qubits(self):
        graph = build_interaction_graph(
            4,
            {
                (0, 1): {"cx": 2, "cz": 1},
                (1, 3): {"unknown_gate": 2},
            },
        )

        self.assertEqual(set(graph.nodes), {"q0", "q1", "q2", "q3"})
        self.assertEqual(graph["q0"]["q1"]["weight"], 8)
        self.assertEqual(graph["q1"]["q3"]["weight"], 2)
        self.assertEqual(graph.number_of_edges(), 2)

    def test_parse_qiskit_counts_two_qubit_gates_and_ignores_single_qubit_gates(self):
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        circuit.cz(1, 2)

        graph = parse_qiskit(circuit)

        self.assertEqual(graph["q0"]["q1"]["weight"], 6)
        self.assertEqual(graph["q1"]["q2"]["weight"], 2)
        self.assertNotIn("q0", graph["q2"])

    def test_parse_qiskit_warns_and_connects_all_pairs_for_multi_qubit_gates(self):
        circuit = QuantumCircuit(3)
        circuit.ccx(0, 1, 2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = parse_qiskit(circuit)

        self.assertTrue(any("Decomposing into all qubit pairs" in str(w.message) for w in caught))
        self.assertEqual(graph.number_of_edges(), 3)
        self.assertTrue(all(data["weight"] == 1 for _, _, data in graph.edges(data=True)))

    def test_parse_qasm_builds_weighted_interaction_graph(self):
        qasm = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
'''

        graph = parse_qasm(qasm)

        self.assertEqual(graph["q0"]["q1"]["weight"], 3)


if __name__ == "__main__":
    unittest.main()
