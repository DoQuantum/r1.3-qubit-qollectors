# Qubit-to-QuQuart Circuit Mapper

A Python implementation using QuTiP that converts qubit circuits (2-level systems) into qu-quart circuits (4-level systems) while minimizing cross-qu-quart gate operations through intelligent qubit pairing optimization.

## Overview

This tool addresses the challenge of mapping quantum circuits designed for qubit systems onto qu-quart (ququart) hardware. The key optimization objective is to minimize **cross-qu-quart gates**, which introduce computational overhead and errors.

### Key Features

- **Best Mapping Optimization (BMO)**: Automatically finds optimal qubit pairing to minimize cross-qu-quart interactions
- **Flexible Mapping**: Support for both automatic optimization and manual mapping specification
- **Comprehensive Analysis**: Tracks and reports cross-qu-quart gate counts and optimization metrics
- **QuTiP Integration**: Built on the QuTiP quantum toolbox for robust quantum operations

## Installation

### Prerequisites

```bash
# Create a virtual environment (recommended)
python -m venv qudit
source qudit/bin/activate  # On Windows: qudit\Scripts\activate

# Install required packages
pip install qutip numpy
```

## Quick Start

### Basic Usage

```python
from qubit_to_ququart_mapper import QubitCircuit, qubit_to_ququart_circuit
import numpy as np

# Create a 4-qubit circuit
circuit = QubitCircuit(num_qubits=4)

# Define gates
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

# Add circuit layers
circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])
circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])

# Convert to qu-quart circuit (automatic optimization)
ququart_circuit = qubit_to_ququart_circuit(circuit)

print(f"Mapping: {ququart_circuit.mapping}")
print(f"Cross-qu-quart gates: {ququart_circuit.cross_ququart_gate_count}")
```

### Manual Mapping

```python
# Specify a custom qubit pairing
custom_mapping = [(0, 2), (1, 3)]  # Pair q0 with q2, q1 with q3
ququart_circuit = qubit_to_ququart_circuit(circuit, mapping=custom_mapping)
```

## Architecture

### Data Structures

#### QubitCircuit
Represents a qubit circuit as a list of layers.

```python
circuit = QubitCircuit(num_qubits=4)
circuit.add_layer([
    (gate_matrix, [qubit_indices])
])
```

Each layer contains gates that can act on different qubits simultaneously.

#### QuQuartCircuit
Represents the converted qu-quart circuit.

```python
# Automatically created by conversion function
ququart_circuit = qubit_to_ququart_circuit(qubit_circuit)

# Access properties
print(ququart_circuit.num_ququarts)              # Number of qu-quarts
print(ququart_circuit.mapping)                   # Qubit pairing
print(ququart_circuit.cross_ququart_gate_count)  # Optimization metric
```

### Core Functions

#### 1. Best Mapping Optimization (BMO)

```python
mapping = best_mapping_optimization(qubit_circuit)
```

**Algorithm:**
1. Build weighted interaction graph
   - Nodes: Individual qubits (q0-q3)
   - Edge weights: Count of 2-qubit gates between each pair
2. Evaluate all possible pairings for 4 qubits:
   - Option A: (q0,q1) + (q2,q3)
   - Option B: (q0,q2) + (q1,q3)
   - Option C: (q0,q3) + (q1,q2)
3. Select pairing that minimizes cross-qu-quart gates

**Cost Function:**
Minimize the number of 2-qubit gates that connect qubits in different qu-quarts.

#### 2. Gate Conversion

The conversion handles two types of gates:

**Single-Qubit Gates:**
- Embedded into 4×4 qu-quart representation
- If qubit is in lower level (pos=0): Gate ⊗ I₂
- If qubit is in upper level (pos=1): I₂ ⊗ Gate

**Two-Qubit Gates:**
- **Within same qu-quart**: Directly becomes single qu-quart gate (4×4 matrix)
- **Across qu-quarts**: Flagged as cross-qu-quart gate (affects optimization metric)

#### 3. Main Conversion Function

```python
ququart_circuit = qubit_to_ququart_circuit(qubit_circuit, mapping=None)
```

**Parameters:**
- `qubit_circuit`: Input QubitCircuit object
- `mapping`: Optional manual mapping (uses BMO if None)

**Returns:**
- QuQuartCircuit with converted gates and optimization metrics

## Examples

### Example 1: High Interaction Circuit

```python
# Circuit with strong coupling between specific pairs
circuit = QubitCircuit(num_qubits=4)

# Many gates between (q0,q1) and (q2,q3)
for _ in range(10):
    circuit.add_layer([(CNOT, [0, 1]), (CNOT, [2, 3])])

ququart = qubit_to_ququart_circuit(circuit)
# Expected: mapping [(0,1), (2,3)] with minimal cross-qu-quart gates
```

### Example 2: Comparing Mappings

```python
# Test different mappings
mappings = [
    [(0, 1), (2, 3)],
    [(0, 2), (1, 3)],
    [(0, 3), (1, 2)]
]

for mapping in mappings:
    ququart = qubit_to_ququart_circuit(circuit, mapping=mapping)
    print(f"{mapping}: {ququart.cross_ququart_gate_count} cross-gates")
```

### Example 3: Edge Cases

```python
# Circuit with only single-qubit gates
circuit = QubitCircuit(num_qubits=4)
H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
circuit.add_layer([(H, [i]) for i in range(4)])

ququart = qubit_to_ququart_circuit(circuit)
# Result: 0 cross-qu-quart gates (any mapping is equivalent)
```

## Running Tests

```bash
# Run all unit tests
python test_mapper.py

# Run comprehensive examples
python example_usage.py

# Run basic demo
python qubit_to_ququart_mapper.py
```

### Test Coverage

The test suite includes:
- ✓ Circuit creation and validation
- ✓ BMO correctness for various topologies
- ✓ Single-qubit gate conversion
- ✓ Within-qu-quart 2-qubit gates
- ✓ Cross-qu-quart 2-qubit gates
- ✓ Mixed layers
- ✓ Optimization benefit verification
- ✓ Edge cases (no 2-qubit gates, isolated qubits)

## Performance Metrics

### Optimization Example

For a circuit with:
- 20 gates within pairs (q0,q1) and (q2,q3)
- 1 gate crossing pairs

**Results:**
- Optimal mapping [(0,1), (2,3)]: **1 cross-qu-quart gate**
- Suboptimal mapping [(0,2), (1,3)]: **20 cross-qu-quart gates**

**Improvement: 95% reduction in cross-qu-quart gates**

## Theory Background

### Why Minimize Cross-Qu-Quart Gates?

Cross-qu-quart gates are expensive because they:
1. Require interaction between separate 4-level systems
2. Introduce additional decoherence and error
3. Increase circuit depth and execution time
4. May require additional hardware resources

### Qubit to Qu-Quart Mapping

A qu-quart is a 4-level quantum system (d=4) that can encode 2 qubits:
- Qu-quart levels: |0⟩, |1⟩, |2⟩, |3⟩
- Qubit mapping:
  - Lower qubit: |0⟩↔|0⟩, |1⟩↔|1⟩
  - Upper qubit: |0⟩↔|2⟩, |1⟩↔|3⟩

### Gate Dimension Conversion

- **Qubit gate**: 2×2 matrix
- **2-qubit gate**: 4×4 matrix
- **Qu-quart gate**: 4×4 matrix
- **2-qu-quart gate**: 16×16 matrix

## Limitations and Future Work

### Current Limitations

1. **4-Qubit Restriction**: BMO currently only supports 4-qubit systems
2. **Static Mapping**: Mapping is determined once at the beginning
3. **Simplified Cross-Qu-Quart Representation**: Full tensor product embedding for cross-qu-quart gates not yet implemented

### Potential Extensions

1. **Scalability**: Extend BMO to n-qubit systems
2. **Dynamic Remapping**: Allow mapping changes during circuit execution
3. **Hardware-Aware Optimization**: Incorporate actual qu-quart hardware constraints
4. **Advanced Cost Functions**: Include gate fidelity, execution time, and error rates
5. **Hybrid Optimization**: Combine BMO with other optimization techniques (e.g., circuit rewriting)

## File Structure

```
r1.3-qubit-qollectors/
├── qubit_to_ququart_mapper.py    # Main implementation
├── test_mapper.py                 # Unit tests
├── example_usage.py               # Comprehensive examples
├── MAPPER_README.md               # This file
└── README.md                      # General project info
```

## API Reference

### Classes

#### QubitCircuit
```python
QubitCircuit(num_qubits: int)
    .add_layer(gates: List[Tuple[np.ndarray, List[int]]])
    .layers: List
    .num_qubits: int
```

#### QuQuartCircuit
```python
QuQuartCircuit(num_ququarts: int, mapping: List[Tuple[int, int]])
    .add_layer(gates: List[Tuple[np.ndarray, List[int]]], is_cross_ququart: bool)
    .layers: List
    .num_ququarts: int
    .mapping: List[Tuple[int, int]]
    .cross_ququart_gate_count: int
```

### Functions

#### best_mapping_optimization
```python
best_mapping_optimization(qubit_circuit: QubitCircuit) -> List[Tuple[int, int]]
```
Finds optimal qubit pairing to minimize cross-qu-quart gates.

#### qubit_to_ququart_circuit
```python
qubit_to_ququart_circuit(
    qubit_circuit: QubitCircuit,
    mapping: List[Tuple[int, int]] = None
) -> QuQuartCircuit
```
Main conversion function from qubit to qu-quart representation.

#### get_ququart_index
```python
get_ququart_index(
    qubit_idx: int,
    mapping: List[Tuple[int, int]]
) -> Tuple[int, int]
```
Determines which qu-quart a qubit belongs to and its position.

#### create_example_circuit
```python
create_example_circuit() -> QubitCircuit
```
Creates a sample 4-qubit circuit for testing and demonstration.

## Contributing

This is a research project for quantum circuit optimization. Contributions and suggestions are welcome!

## References

- **QuTiP Documentation**: https://qutip.org/
- **Quantum Circuit Optimization**: Standard techniques for reducing gate counts and improving fidelity
- **Qudit Computing**: Extension of qubit computing to d-level systems

## License

This code is provided for research and educational purposes.

---

**Author:** Quantum Research Team
**Date:** January 2026
**Version:** 1.0
