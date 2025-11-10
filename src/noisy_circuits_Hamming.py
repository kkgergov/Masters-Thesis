import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt
from src.utils import create_noise_model

def simulate_with_noise_3D(circuit, true_output, noise_models, max_shots=1000, step=100):

    """
    Simulate noisy circuit and compute average Hamming distance vs shots for every noise level
    1. circuits: list of [QuantumCircuit, num_qubits]
    2. true_outputs: list of expected ideal output strings for each circuit
    3. circuit_index: index of the circuit to simulate
    4. noise_models: list of NoiseModel objects for different noise levels
    5. max_shots: maximum number of shots to simulate
    6. step: step size for shots
    7. Returns: shot_counts, hamming_distances
    """
    
    all_hamming_distances = []
    all_shot_counts = []

    for noise_model in noise_models:
        
        simulator = AerSimulator(noise_model=noise_model)
        transpiled_circ = transpile(circuit, simulator, optimization_level=0)
        
        shot_counts = range(step, max_shots + 1, step)
        hamming_distances = []
        
        for shots in shot_counts:
            # Execute simulation
            job = simulator.run(transpiled_circ, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Calculate Hamming distance
            avg_hamming = calculate_hamming_distance(counts, true_output, shots)
            hamming_distances.append(avg_hamming)

        all_hamming_distances.append(hamming_distances)
        all_shot_counts = shot_counts  # Same for all noise levels

    return all_shot_counts, all_hamming_distances

def simulate_noise_range(circuit, true_output, noise_models, shots=1000):
    """
    Simulate noisy circuit over a range of noise models and compute average Hamming distances
    """
    hamming_distances = []

    for noise_model in noise_models:
        simulator = AerSimulator(noise_model=noise_model)
        transpiled_circ = transpile(circuit, simulator, optimization_level=0)

        job = simulator.run(transpiled_circ, shots=shots)
        result = job.result()
        counts = result.get_counts()

        avg_hamming = calculate_hamming_distance(counts, true_output, shots)
        hamming_distances.append(avg_hamming)

    return hamming_distances

def calculate_hamming_distance(counts, true_output, total_shots):
    """
    Calculate average Hamming distance between measurements and true output
    """
    total_distance = 0
    
    for output_string, count in counts.items():
        # Calculate Hamming distance for this output
        distance = sum(1 for a, b in zip(output_string, true_output) if a != b)
        total_distance += distance * count
    
    return total_distance / total_shots

def create_circuits():
    
    circuits = []
    true_outputs = []

    """Circuit 0: Single qubit with X gate"""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    circuits.append([qc, 1])
    true_outputs.append("1")

    """Circuit 1: Single qubit with 5 X gate"""
    qc = QuantumCircuit(1, 1)
    for _ in range(5):
        qc.x(0)
    qc.measure(0, 0)
    circuits.append([qc, 1])
    true_outputs.append("0")

    """Circuit 2: Three qubits with single X gate"""
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.x(1)
    qc.x(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append([qc, 3])
    true_outputs.append("111")

    """Circuit 3: Three qubits with 5 X gates"""
    qc = QuantumCircuit(3, 3)
    for _ in range(5):
        qc.x(0)
    for _ in range(5):
        qc.x(1)
    for _ in range(5):
        qc.x(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append([qc, 3])
    true_outputs.append("111")

    """Circuit 4: CX Experiment 1"""
    qc = QuantumCircuit(3, 3)

    qc.x(0)
    for _ in range(5):
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(2, 0)

    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append([qc, 3])
    true_outputs.append("111")

    """Circuit 5: CX Experiment 2"""
    qc = QuantumCircuit(3, 3)

    qc.x(0)
    for _ in range(5):
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append([qc, 3])
    true_outputs.append("101")

    return circuits, true_outputs

def create_noise_models(noise_levels):

    noise_models = []
    for level in noise_levels:
        noise_model = create_noise_model(gate_error=level, measurement_error=level)
        noise_models.append(noise_model)

    return noise_models

def plot_3d(circuit_index):
    
    circuits, true_outputs = create_circuits()
    circuit, true_output = circuits[circuit_index][0], true_outputs[circuit_index]

    noise_levels = np.linspace(0, 1, 101) # 0% to 100% error rates
    noise_models = create_noise_models(noise_levels)


    all_shot_counts, all_hamming_distances = simulate_with_noise_3D(
        circuit, true_output, noise_models=noise_models, max_shots=1000, step=50,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(all_shot_counts, noise_levels)
    Z = np.array(all_hamming_distances)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Number of Shots')
    ax.set_ylabel('Noise Level')
    ax.set_zlabel('Average Hamming Distance')
    ax.set_title(f'Hamming Distance vs Shots and Noise Level (Circuit {circuit_index})')

    plt.savefig(f'3D_plot_circuit_{circuit_index}.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_circuits(circuit_idx1, circuit_idx2, shots=10000):
    circuits, true_outputs = create_circuits()
    
    circuit1, true_output1 = circuits[circuit_idx1][0], true_outputs[circuit_idx1]
    circuit2, true_output2 = circuits[circuit_idx2][0], true_outputs[circuit_idx2]

    noise_levels = np.linspace(0, 0.3, 101) # 0% to 30% error rates
    noise_models = create_noise_models(noise_levels)

    hamming_distances1 = simulate_noise_range(circuit1, true_output1, noise_models, shots)
    hamming_distances2 = simulate_noise_range(circuit2, true_output2, noise_models, shots)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, hamming_distances1, label=f'Circuit {circuit_idx1}')
    plt.plot(noise_levels, hamming_distances2, label=f'Circuit {circuit_idx2}')
    plt.xlabel('Noise Level')
    plt.ylabel('Average Hamming Distance')
    plt.title(f'Comparison of Circuits {circuit_idx1} and {circuit_idx2} at {shots} Shots')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'compare_circuits_{circuit_idx1}_vs_{circuit_idx2}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    
    # circuit_index = 5
    # plot_3d(circuit_index)

    compare_circuits(4, 5, shots=30000)