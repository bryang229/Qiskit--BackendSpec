from qiskit import QuantumCircuit, transpile
from backendspec import BackendSpec

spec = BackendSpec()

qc = QuantumCircuit(2,2)
qc.x(0)
qc.cx(0,1)
qc.h(1)


spec.increase_qubits(10, 'hexagonal')
spec.set_gate_property('id', 'gate_error', 0, .01, True)
spec.set_gate_properties_distribution('id', 'gate_error', .001, .3, True)
spec.swap_basis_gate('x', 'y', True)
spec.swap_2q_basis_gate_distribution('cx','ecr',[.01, .1], [1e-7]*2)
spec.scale_gate_property('y', 'gate_error', 2)



# backend = spec.new_backend()
# print(backend.gates_df.loc[backend.gates_df.gate == 'id'])

# transpiled = transpile(qc, backend, basis_gates=spec.basis_gates)
# print(transpiled)
# print(spec.qubit_properties)
