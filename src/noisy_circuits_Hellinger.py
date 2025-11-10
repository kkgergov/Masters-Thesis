import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt

from itertools import product
from src.utils import create_noise_model

def hellinger_distance(p, q):
    """
    Calculate Hellinger distance between two probability distributions.
    
    Parameters:
    p, q: arrays representing probability distributions
    
    Returns:
    float: Hellinger distance between p and q
    """
    return np.sqrt(1 - np.sum(np.sqrt(p * q)))

def create_circuits():

    circuits = []
    true_dists = []

    """Circuit 0: Three qubits with single H gate"""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append([qc, 3])
    true_dists.append({
        '000': 1/8,
        '111': 1/8,
        '001': 1/8,
        '110': 1/8,
        '010': 1/8,
        '101': 1/8,
        '011': 1/8,
        '100': 1/8
    })

    """Circuit 1: Three qubits with 5 H gates"""
    qc = QuantumCircuit(3, 3)
    for _ in range(5):
        qc.h(0)
    for _ in range(5):
        qc.h(1)
    for _ in range(5):
        qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append([qc, 3])
    true_dists.append({
        '000': 1/8,
        '111': 1/8,
        '001': 1/8,
        '110': 1/8,
        '010': 1/8,
        '101': 1/8,
        '011': 1/8,
        '100': 1/8
    })

    """Circuit 2: GHZ State (|000⟩ + |111⟩)/√2"""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append([qc, 3])
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

def simulate_circuit(circuit_idx = 0, noise_level=0.01, shots=10000):
    """
    Simulate Bell pair generation with noise and calculate Hellinger distance
    """
    circuits, ideal_dists = create_circuits()
    circuit, ideal_dist = circuits[circuit_idx][0], ideal_dists[circuit_idx]
    noise_model = create_noise_model(gate_error=noise_level, 
                                   measurement_error=noise_level)
    
    # Simulate with noise
    simulator = AerSimulator(noise_model=noise_model)
    result = simulator.run(circuit, shots=shots).result()
    noisy_counts = result.get_counts()
    
    # Generate possible states
    all_states = binary_vectors = [''.join(bits) for bits in product('01', repeat=3)]
    
    # Ideal distribution as array
    p_ideal = np.array([ideal_dist.get(state, 0.0) for state in all_states])
    
    # Noisy distribution as array
    total_shots = sum(noisy_counts.values())
    p_noisy = np.array([noisy_counts.get(state, 0) / total_shots for state in all_states])
    
    # Calculate Hellinger distance
    h_dist = hellinger_distance(p_ideal, p_noisy)
    
    return h_dist, noisy_counts, ideal_dist

def hellinger_vs_noise_level(circuit_index=0, max_noise=0.1, steps=20, shots=10000):
    """
    Calculate Hellinger distance for increasing noise levels
    """
    noise_levels = np.linspace(0, max_noise, steps)
    hellinger_distances = []
    
    for noise in noise_levels:
        h_dist, _, _ = simulate_circuit(circuit_idx=circuit_index, noise_level=noise, shots=shots)
        hellinger_distances.append(h_dist)
        print(f"Noise level: {noise:.3f}, Hellinger distance: {h_dist:.4f}")
    
    return noise_levels, hellinger_distances

def hellinger_vs_shots(circuit_index=0, noise_level=0.05, max_shots=10000, step=500):
    """
    Calculate Hellinger distance for increasing number of shots
    """
    shot_counts = range(step, max_shots + 1, step)
    hellinger_distances = []
    
    for shots in shot_counts:
        h_dist, _, _ = simulate_circuit(circuit_idx=circuit_index, noise_level=noise_level, shots=shots)
        hellinger_distances.append(h_dist)
    
    return shot_counts, hellinger_distances

def plot_3d(circuit_index):
    """
    Plot 3D surface of Hellinger distance vs noise level and shots
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    noise_levels = np.linspace(0, 0.4, 100)
    shot_counts = np.arange(10, 1000, 100)
    
    H = np.zeros((len(noise_levels), len(shot_counts)))
    
    for i, noise in enumerate(noise_levels):
        for j, shots in enumerate(shot_counts):
            h_dist, _, _ = simulate_circuit(circuit_idx=circuit_index, noise_level=noise, shots=shots)
            H[i, j] = h_dist
    
    X, Y = np.meshgrid(shot_counts, noise_levels)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, H, cmap=cm.viridis)
    ax.set_xlabel('Number of Shots')
    ax.set_ylabel('Noise Level')
    ax.set_zlabel('Hellinger Distance')
    fig.colorbar(surf)
    plt.title('Hellinger Distance vs Noise Level and Shots')
    plt.show()

# Example usage and visualization
if __name__ == "__main__":
    # Single simulation example
    circuit_index = 2

    # plot_3d(circuit_index)

    # h_dist, noisy_counts, ideal_dist = simulate_circuit(circuit_idx=circuit_index, noise_level=0.05, shots=10000)
    # print("Ideal distribution:", ideal_dist)
    # print("Noisy distribution:", {k: v/10000 for k, v in noisy_counts.items()})
    # print(f"Hellinger distance: {h_dist:.4f}")
    
    # Plot Hellinger distance vs noise level for 10, 30, 100 and 10000 shots and max noise 0.4 and plot them together as lines
    # noise_levels1, h_distances1 = hellinger_vs_noise_level(circuit_index=circuit_index, max_noise=0.4, steps=20, shots=10)
    # noise_levels2, h_distances2 = hellinger_vs_noise_level(circuit_index=circuit_index, max_noise=0.4, steps=20, shots=30)
    # noise_levels3, h_distances3 = hellinger_vs_noise_level(circuit_index=circuit_index, max_noise=0.4, steps=20, shots=100)
    # noise_levels4, h_distances4 = hellinger_vs_noise_level(circuit_index=circuit_index, max_noise=0.4, steps=20, shots=10000)

    # plt.figure(figsize=(10, 6))
    # plt.plot(noise_levels1, h_distances1, label='10 shots')
    # plt.plot(noise_levels2, h_distances2, label='30 shots')
    # plt.plot(noise_levels3, h_distances3, label='100 shots')
    # plt.plot(noise_levels4, h_distances4, label='10000 shots')
    # plt.xlabel('Noise Level')
    # plt.ylabel('Hellinger Distance')
    # plt.title('Hellinger Distance vs Noise Level for Different Shot Counts')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Plot Hellinger distance and hellinger distance standart deviation vs number of shots
    shot_counts, h_distances_shots = hellinger_vs_shots(circuit_index=circuit_index, noise_level=0.45, max_shots=2000, step=10)

    plt.figure(figsize=(10, 6))
    plt.plot(shot_counts, h_distances_shots, label=f'Noise level 0.25')
    plt.xlabel('Number of Shots')
    plt.ylabel('Hellinger Distance')
    plt.title('Hellinger Distance vs Number of Shots')
    plt.legend()
    plt.grid(True)
    plt.show()