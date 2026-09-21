from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import math


# =========================
# SETTINGS
# =========================

num_qubits = 1
shots = 1000

gate = "H"

# Used by parameterized gates such as RX, RY, RZ, and P
angle = math.pi / 2


# =========================
# CREATE CIRCUIT
# =========================

qc = QuantumCircuit(num_qubits)

if gate == "H":
    qc.h(0)

elif gate == "X":
    qc.x(0)

elif gate == "Y":
    qc.y(0)

elif gate == "Z":
    qc.z(0)

elif gate == "RX":
    qc.rx(angle, 0)

elif gate == "RY":
    qc.ry(angle, 0)

elif gate == "RZ":
    qc.rz(angle, 0)

elif gate == "P":
    qc.p(angle, 0)

else:
    print("Unknown gate:", gate)
    exit()


# Add measurement
qc.measure_all()


# =========================
# DISPLAY CIRCUIT
# =========================

print("\nQuantum Circuit:")
# print(qc)


# =========================
# RUN QRNG
# =========================

simulator = AerSimulator()

job = simulator.run(
    qc,
    shots=shots,
    memory=True
)

result = job.result()


# =========================
# GET RANDOM BITS
# =========================

bits = result.get_memory()

random_string = ''.join(bits)


# =========================
# DISPLAY RESULTS
# =========================

print("\nRandom bit string:")
print(random_string)

print("\nLength:", len(random_string))

print("\nCounts:")
print(result.get_counts())