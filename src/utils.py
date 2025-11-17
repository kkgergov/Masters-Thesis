# Core
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import Statevector

# Misc utils
from itertools import product
from functools import partial

# Visualization
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Checkbox, Dropdown

def calculate_hamming_distance(noisy_counts, true_output, total_shots):
    """
    Calculate average Hamming distance between measurements and true output.

    Parameters:
    noisy_counts: dict, noisy measurement outcomes
    true_output: str, ideal measurement outcome
    total_shots: int, total number of shots

    Returns:
    float: average Hamming distance
    """
    total_distance = 0

    for output_string, count in noisy_counts.items():
        distance = sum(1 for a, b in zip(output_string, true_output) if a != b)
        total_distance += distance * count
    
    return total_distance / total_shots

def get_ideal_dist(qc: QuantumCircuit):
    # Use the statevector simulator to get the ideal statevector
    simulator = AerSimulator(method='statevector')
    compiled_circuit = transpile(qc, simulator)
    statevector = Statevector(compiled_circuit)

    # Calculate probabilities from the statevector
    probabilities = np.abs(statevector)**2

    # Generate basis state labels (e.g., '00', '01', '10', '11')
    num_qubits = qc.num_qubits
    basis_states = [format(i, '0'+str(num_qubits)+'b') for i in range(2**num_qubits)]

    # Create the dictionary: {basis_state: probability}
    ideal_distribution = dict(zip(basis_states, probabilities))

    return ideal_distribution

def calculate_hellinger_distance(noisy_counts, ideal_dist, total_shots, n_qubits):
    """
    Calculate Hellinger distance between mesurements and ideal distribution.
    
    Parameters:
    noisy_counts: dict, noisy measurement outcomes
    ideal_dist: dict, ideal measurement probabilities
    total_shots: int, total number of shots
    n_qubits: int, number of qubits

    Returns:
    float: Hellinger distance
    """

    # Generate all basis states in ordered manner
    all_states = binary_vectors = [''.join(bits) for bits in product('01', repeat=n_qubits)]

    # Noisy distribution as array
    p = np.array([noisy_counts.get(state, 0) / total_shots for state in all_states])

    # Ideal distribution as array
    q = np.array([ideal_dist.get(state, 0.0) for state in all_states])

    return np.sqrt(1 - np.sum(np.sqrt(p * q)))

def create_noise_model(gate_error=0.01, measurement_error=0.01):
    """
    Create a noise model with depolarizing and measurement errors
    Parameters:
    gate_error: float, depolarizing error rate for gates
    measurement_error: float, depolarizing error rate for measurements
    Returns:
    NoiseModel: Qiskit noise model
    """
    noise_model = NoiseModel()
    
    # Apply to comprehensive list of gates
    all_single_qubit_gates = [
        # Pauli gates
        'x', 'y', 'z', 'id',
        # Clifford gates
        'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg',
        # Rotation gates
        'rx', 'ry', 'rz', 'p', 'u1', 'u2', 'u3', 'u'
    ]
    
    all_two_qubit_gates = [
        'cx', 'cy', 'cz', 'cp', 'crx', 'cry', 'crz',
        'cu1', 'cu2', 'cu3', 'swap', 'iswap', 'dcx',
        'ecr', 'rxx', 'ryy', 'rzz'
    ]

    # Add depolarizing error to single-qubit gates
    single_qubit_error = depolarizing_error(gate_error, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_error, all_single_qubit_gates)

    # Add depolarizing error to two-qubit gates
    two_qubit_error = depolarizing_error(gate_error, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, all_two_qubit_gates)

    # Add measurement error
    measurement_error_model = depolarizing_error(measurement_error, 1)
    noise_model.add_all_qubit_quantum_error(measurement_error_model, 'measure')
    
    return noise_model

def simulate_with_noise_3D(circuit, ideal_output, noise_levels = np.linspace(0, 0.3, 101), shot_counts = range(100, 10001, 100), distance_type='hamming'):

    """
    Simulate noisy circuit and compute average Hamming distance vs shots vs noise level
    1. circuit: list of [QuantumCircuit, num_qubits]
    2. ideal_output: expected ideal output of the circuit (either distribution or a basis state)
    3. noise_levels: list of noise levels to simulate, default: 100 levels from [0, 0.3]
    4. shot_counts: list of shot counts to simulate, default: 100 counts from [100, 10000]
    5. distance_type: type of distance metric to use (e.g., 'hamming')
    6. Returns: 2D array of distances with shape noise_levels x shot_counts
    """

    print("Simulating circuit with noise range [{}, {}] and shot counts [{}, {}]".format(noise_levels[0], noise_levels[-1], shot_counts[0], shot_counts[-1]))
    print("Using distance metric: {}".format(distance_type))

    if distance_type == 'hamming':
        distance_calc_func = calculate_hamming_distance
    elif distance_type == 'hellinger':
        distance_calc_func = partial(calculate_hellinger_distance, n_qubits=circuit.num_qubits)  
    else:
        raise ValueError("Unsupported distance type. Use 'hamming' or 'hellinger'.")  

    # Create noise models for each noise level
    noise_models = []
    for level in noise_levels:
        noise_model = create_noise_model(gate_error=level, measurement_error=level)
        noise_models.append(noise_model)

    H = np.zeros((len(noise_models), len(shot_counts)))

    for i, noise_model in enumerate(noise_models):

        simulator = AerSimulator(noise_model=noise_model)
        transpiled_circ = transpile(circuit, simulator, optimization_level=0)

        for j, shots in enumerate(shot_counts):
            # Execute simulation
            job = simulator.run(transpiled_circ, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Calculate distance
            H[i, j] = distance_calc_func(counts, ideal_output, shots)

    return H

def plot_3d(distance_data, noise_levels, shot_counts):
    """Plot 3D surface of distance data vs noise levels and shot counts"""

    X, Y = np.meshgrid(shot_counts, noise_levels)
    Z = distance_data

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Number of Shots')
    ax.set_ylabel('Noise Level')
    ax.set_zlabel('Hamming Distance')
    ax.set_title('Hamming Distance vs Noise Level and Number of Shots')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

class DistanceVisualizer:
    def __init__(self, shots_array, noise_intensities, hamming_data, hellinger_data, n_qubits):
        self.shots_array = shots_array
        self.noise_intensities = noise_intensities
        self.theoretical_max_Hamming = n_qubits / 2

        # Hamming data is of the format (experiments, noise_levels, shot_counts)
        self.hamming_data = hamming_data

        # Hellinger data to compare with the hamming data
        self.hellinger_data = hellinger_data

        # Precompute statistics mean and std across experiments for each noise level and shot count
        self.mean_hamming = np.mean(self.hamming_data, axis=0)
        self.std_hamming = np.std(self.hamming_data, axis=0)

    def plot_interactive(self, run_index, noise_index):
        hellinger_slice = self.hellinger_data[noise_index, :]
        hamming_slice = self.hamming_data[run_index, noise_index, :]
        mean_slice = self.mean_hamming[noise_index, :]
        std_slice = self.std_hamming[noise_index, :]

        fig, ax = plt.subplots(figsize=(12, 7))

        # Display std centered around mean
        ax.fill_between(self.shots_array, 
                        mean_slice - std_slice, 
                        mean_slice + std_slice, 
                        color='gray', alpha=0.3, label='Mean ± Std Dev')

        # Display Hellinger distance
        ax.plot(self.shots_array, hellinger_slice, color='r')

        # Display chosen Hamming run 
        ax.plot(self.shots_array, hamming_slice, color='b')

        # Display Hellinger std
        ax.plot(self.shots_array, std_slice, color='y')

        # Display y up to theoretical max Hamming distance
        ax.set_ylim(0, self.theoretical_max_Hamming + 0.1)

        # Customize plot
        ax.set_xlabel('Number of Shots', fontsize=12)
        ax.set_ylabel('Hamming Distance', fontsize=12)
        actual_noise = self.noise_intensities[noise_index]
        ax.set_title(f'Hamming Distance vs Number of Shots\nNoise Intensity: {actual_noise:.4f}', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Plot theoretical max Hamming distance
        ax.axhline(y=self.theoretical_max_Hamming, color='r', linestyle='--', label='Theoretical Max Hamming Distance')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
    def create_dashboard(self):
        """Create a comprehensive interactive dashboard"""
        @interact
        def dashboard(
            noise_intensity=FloatSlider(
                min=min(self.noise_intensities),
                max=max(self.noise_intensities),
                step=(max(self.noise_intensities)-min(self.noise_intensities))/len(self.noise_intensities),
                value=np.median(self.noise_intensities),
                description='Noise Intensity:',
                continuous_update=True
            ),
            run_index=Dropdown(
                options=[(f'Experiment {i}', i) for i in range(self.hamming_data.shape[0])],
                value=0,
                description='Run:'
            )
        ):
            noise_idx = np.argmin(np.abs(self.noise_intensities - noise_intensity))
            self.plot_interactive(run_index, noise_idx)
