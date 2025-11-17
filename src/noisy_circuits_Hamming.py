import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.circuit.library import n_local

from utils import simulate_with_noise_3D
from utils import plot_3d

def basic_circuits():
    
    circuits = []
    true_outputs = []

    """Circuit 0: Single qubit with X gate"""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    circuits.append(qc)
    true_outputs.append("1")

    """Circuit 1: Single qubit with 5 X gate"""
    qc = QuantumCircuit(1, 1)
    for _ in range(5):
        qc.x(0)
    qc.measure(0, 0)
    circuits.append(qc)
    true_outputs.append("0")

    """Circuit 2: Three qubits with two X gates"""
    qc = QuantumCircuit(3, 3)
    for _ in range(2):
        qc.x(0)
    for _ in range(2):
        qc.x(1)
    for _ in range(2):
        qc.x(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc)
    true_outputs.append("000")

    """Circuit 3: CX Experiment 1"""
    qc = QuantumCircuit(3, 3)

    qc.x(0)
    for _ in range(5):
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(2, 0)

    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc)
    true_outputs.append("111")

    """Circuit 4: CX Experiment 2"""
    qc = QuantumCircuit(3, 3)

    qc.x(0)
    for _ in range(5):
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc)
    true_outputs.append("101")

    return circuits, true_outputs

# if __name__ == "__main__":
    
#     # Single simulation example
#     circuit_index = 1

#     circuits, true_outputs = hamming_circuits()
#     circuit, true_output = circuits[circuit_index], true_outputs[circuit_index]

#     noise_levels = np.linspace(0, 0.2, 21)

#     # Create ranges for each segment
#     ranges = [
#         np.arange(10, 250, 10),      # Usually big std reduction occurs here so we want to most precision
#         np.arange(250, 1000, 50),   # Here it slowly starts to converge
#         np.arange(1000, 2000, 100), # Usually completely converges in this range
#         np.arange(2000, 10001, 500)
#     ]
#     shot_counts = np.concatenate(ranges)

#     H = simulate_with_noise_3D(
#         circuit, true_output, noise_levels=noise_levels, shot_counts=shot_counts, distance_type='hamming'
#     )

#     # Visualization
#     plot_3d(H, noise_levels, shot_counts)