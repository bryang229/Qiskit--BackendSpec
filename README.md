# Qiskit BackendSpec Framework

## Overview

Welcome to the Qiskit BackendSpec Framework! This project provides a versatile framework for generating Qiskit backends, specifically designed for testing the transpilation of quantum circuits on experimental backend configurations. The primary goal is to empower users to create customizable backends with experimental features such as modified qubit count, unique coupling map configuration, modified qubit error rates, and more.

## Features

- **Customizable BackendSpec**: Generate a Qiskit BackendSpec object, which serves as a template for the generation of Backend objects. The user is able to modify the features of the BackendSpec and sample from this data for the generation of new backends.
  
- **Qubit Count Configuration**: Define the number of qubits for the backend, allowing you to simulate quantum circuits of different sizes.

- **Coupling Map Configuration**: Specify the coupling map for the backend, enabling you to model the connectivity of physical qubits on real or hypothetical quantum devices.

- **Qubit Error Rates**: Set custom error rates for individual qubits.

## Getting Started

### Prerequisites

- Python 3.x
- Qiskit (install via `pip install qiskit`)

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/qiskit_backendspec.git
cd qiskit_backendspec
```
### Usage
1. Import the BackendSpec class from the framework:
```
from qiskit_backendspec import BackendSpec
```
2. Create an instance of the BackendSpec class:
```
backend_spec = BackendSpec()
```
3. Customize the backend features by configuring parameters such as qubit count, coupling map, and error rates:
```
backend_spec.set_qubit_property(0, 't1', 3.7) #set the t1 time for qubit 0
backend_spec.set_coupling_map(CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4]])) #set coupling map
backend_spec.increase_qubits(3, 'square') #increase number of qubits
```
4. Generate the qiskit backend object:
```
custom_backend = backend_spec.new_backend()
```
5. Transpile the generated backend
```
circuit = QuantumCircuit(5)
transpiled_circuit = transpile(circuit, custom_backend)
```
