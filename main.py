from qiskit import QuantumCircuit, transpile
from backendspec import BackendSpec

spec = BackendSpec()

qc = QuantumCircuit(2,2)
qc.x(0)
qc.cx(0,1)
qc.h(1)

backend = spec.new_backend()

transpiled = transpile(qc, backend)

