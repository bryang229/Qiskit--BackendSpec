from qiskit import QuantumCircuit, transpile
from backendspec import BackendSpec

spec = BackendSpec()

qc = QuantumCircuit(2,2)
qc.x(0)
qc.cx(0,1)
qc.h(1)


spec.increase_qubits(10, 'hexagonal')
spec.set_gate_properties_distribution('id', 'gate_error', .001, .3, True)

backend = spec.new_backend()
print(backend.gates_df.loc[backend.gates_df.gate == 'id'])

transpiled = transpile(qc, backend, basis_gates=spec.basis_gates)
print(transpiled)
# print(spec.qubit_properties)
