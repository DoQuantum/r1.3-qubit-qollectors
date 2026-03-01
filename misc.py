import numpy as np
import math 
import qutip as qt
from qubit_to_ququart_mapper import (
    QubitCircuit,
    best_mapping_optimization,
    qubit_to_ququart_circuit,
    create_example_circuit,
    display_ququart_circuit,
    KNOWN_GATES
)


hadamard = [[1/math.sqrt(2), 1/math.sqrt(2)], 
            [1/math.sqrt(2), -1/math.sqrt(2)]]
identity = [[1, 0],
            [0, 1]]

cnot = [[1, 0, 0, 0], 
        [0, 1, 0, 0],
        [0, 0, 0, 1], 
        [0, 0, 1, 0]]

print()
print(np.kron(hadamard, identity)) # H tensor I --> constant tensor matrix

cnot1 = np.kron(identity, cnot)
print(np.kron(identity, cnot1)) #CNOT tensor I tensor I

'''

def CNOT_circuit_tester():

    circuit = QubitCircuit(num_qubits = 4)

    circuit.add_layer([
        (KNOWN_GATES["CNOT"], [1, 2]),
        (KNOWN_GATES["CNOT"], [0, 3])
    ])

    circuit.add_layer([
        (KNOWN_GATES["CNOT"], [2, 3]) #Cross QQ Gate
    ])

    return circuit


if __name__ == "__main__":
    print("="*60)
    print("Testing 16x16 CNOT Gates")
    print("="*60)

    # Create example circuit
    print("\n1. Creating example 4-qubit circuit...\n")
    qubit_circuit = CNOT_circuit_tester()
    print(f"   {qubit_circuit}")

    # Convert to qu-quart circuit (BMO will find optimal mapping)
    print("\n2. Converting to qu-quart circuit...")
    ququart_circuit = qubit_to_ququart_circuit(qubit_circuit)

    # Display detailed results
    print("\n3. Displaying qu-quart circuit details...")
    display_ququart_circuit(ququart_circuit)
'''