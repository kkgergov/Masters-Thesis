import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt

from utils import simulate_with_noise_3D
from utils import plot_3d

def basic_hellinger():
    circuits = []
    true_dists = []

    """Circuit 1: GHZ State (|000⟩ + |111⟩)/√2"""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc)
    true_dists.append({
        '000': 0.5,
        '111': 0.5,
        '001': 0.0,
        '010': 0.0,
        '100': 0.0,
        '101': 0.0,
        '110': 0.0
    })

    return circuits, true_dists

# # Example usage and visualization
# if __name__ == "__main__":
#     # Single simulation example
#     circuit_index = 2

#     circuits, true_outputs = hellinger_circuits()
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
#         circuit, true_output, noise_levels=noise_levels, shot_counts=shot_counts, distance_type='hellinger'
#     )

#     # Visualization
#     plot_3d(H, noise_levels, shot_counts)