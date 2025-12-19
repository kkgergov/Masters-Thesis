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
import ruptures as rpt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
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
    def __init__(self, shots_array, noise_intensities, circuit_name, hamming_data, hellinger_data, n_qubits):
        self.shots_array = shots_array
        self.noise_intensities = noise_intensities
        self.circuit_name = circuit_name
        self.theoretical_max_Hamming = n_qubits / 2

        # Hamming data is of the format (experiments, noise_levels, shot_counts)
        self.hamming_data = hamming_data

        # Hellinger data to compare with the hamming data
        self.hellinger_data = hellinger_data

        # Precompute statistics mean and std across experiments for each noise level and shot count
        self.mean_hamming = np.mean(self.hamming_data, axis=0)
        self.std_hamming = np.std(self.hamming_data, axis=0)

    def plot_interactive(self, noise_index):
        hellinger_slice = self.hellinger_data[noise_index, :]
        hamming_mean_slice = self.mean_hamming[noise_index, :]
        hamming_std_slice = self.std_hamming[noise_index, :]

        fig, ax = plt.subplots(figsize=(12, 7))

        # Display std centered around mean
        ax.fill_between(self.shots_array, 
                        hamming_mean_slice - hamming_std_slice, 
                        hamming_mean_slice + hamming_std_slice, 
                        color='gray', alpha=0.3, label='Mean ± Std Dev')

        # Smooth the Hellinger, Hamming, Std curves using Savitzky-Golay filter
        hellinger_slice_smooth = savgol_filter(hellinger_slice, 11, 3)
        hamming_slice_smooth = savgol_filter(hamming_mean_slice, 11, 3)
        std_slice_smooth = savgol_filter(hamming_std_slice, 11, 3)

        # Display Hellinger distance
        ax.plot(self.shots_array, hellinger_slice_smooth, color='r')

        # Display chosen Hamming run 
        ax.plot(self.shots_array, hamming_slice_smooth, color='b')

        # Display Hellinger std
        ax.plot(self.shots_array, std_slice_smooth, color='y')

        # Display y up to theoretical max Hamming distance
        ax.set_ylim(0, self.theoretical_max_Hamming + 0.1)

        # Plot legend: Hellinger is red, Hamming is blue, std is yellow
        ax.plot([], [], color='r', label='Hellinger Distance')
        ax.plot([], [], color='b', label='Hamming mean Distance')
        ax.plot([], [], color='y', label='Std Dev')
        ax.legend()

        # Labels and title
        ax.set_xlabel('Number of Shots', fontsize=12)
        actual_noise = self.noise_intensities[noise_index]
        ax.set_title(f'{self.circuit_name}\nNoise Intensity: {actual_noise:.4f}', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Plot theoretical max Hamming distance
        ax.axhline(y=self.theoretical_max_Hamming, color='r', linestyle='--', label='Theoretical Max Hamming Distance')
        ax.legend()

        plt.tight_layout()
        plt.show()
        
    def create_dashboard(self, init_value=None):
        """Create a comprehensive interactive dashboard"""
        @interact
        def dashboard(
            noise_intensity=FloatSlider(
                min=min(self.noise_intensities),
                max=max(self.noise_intensities),
                step=(max(self.noise_intensities)-min(self.noise_intensities))/len(self.noise_intensities),
                value=init_value if init_value is not None else np.median(self.noise_intensities),
                description='Noise Intensity:',
                continuous_update=True
            )
        ):
            noise_idx = np.argmin(np.abs(self.noise_intensities - noise_intensity))
            self.plot_interactive(noise_idx)

class TransitionPointsVisualizer:
    def __init__(self, names_dataset, shots_dataset, noise_dataset, hamming_dataset, hellinger_dataset, n_qubits):
        self.circuit_names = names_dataset
        self.shots_dataset = shots_dataset
        self.noise_dataset = noise_dataset
        self.hamming_dataset = hamming_dataset
        self.hellinger = hellinger_dataset
        self.n_qubits = n_qubits

        self.theoretical_max_Hamming = n_qubits / 2

        # Precompute mean and std for each circuit in the Hamming dataset
        self.mean_hamming = [np.mean(hamming_data, axis=0) for hamming_data in hamming_dataset]
        self.std_hamming = [np.std(hamming_data, axis=0) for hamming_data in hamming_dataset]

        # When noise level makes Hamming mean reach 3.5, save index
        # initialize array with 20
        self.cutoff_indices = np.array([len(noise_dataset[i]) for i in range(len(self.circuit_names))])
        for i in range(len(self.circuit_names)):
            for j in range(len(self.noise_dataset[i])):
                if self.mean_hamming[i][j, -1] >= 3.5:
                    self.cutoff_indices[i] = j
                    break
        

        # Precompute constants to fit Exponential Decay (24, 21, 3)
        self.poly_hellinger_ABC = np.array([
            [fit_exponential_decay_to_data(self.shots_dataset[i], self.hellinger[i][noise_idx, :]) 
             for noise_idx in range(self.hellinger[i].shape[0])]
            for i in range(len(self.circuit_names))
        ])
        self.poly_hamming_std_ABC = np.array([
            [fit_exponential_decay_to_data(self.shots_dataset[i], self.std_hamming[i][noise_idx, :]) 
             for noise_idx in range(self.std_hamming[i].shape[0])]
            for i in range(len(self.circuit_names))
        ])

        # Precompute constants to fit Exponential Decay (24, 21, 81)
        self.poly_hellinger = np.array([
            [exp_decay(self.shots_dataset[i], *self.poly_hellinger_ABC[i][noise_idx]) 
             for noise_idx in range(self.hellinger[i].shape[0])]
            for i in range(len(self.circuit_names))
        ])
        self.poly_hamming_std = np.array([
            [exp_decay(self.shots_dataset[i], *self.poly_hamming_std_ABC[i][noise_idx]) 
             for noise_idx in range(self.std_hamming[i].shape[0])]
            for i in range(len(self.circuit_names))
        ])

    def plot_hellinger(self, noise_index):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, circuit_name in enumerate(self.circuit_names):
            hellinger_slice = self.smoothed_hellinger[i][noise_index, :]
            ax.plot(self.shots_dataset[i], hellinger_slice, label=circuit_name)
            ax.text(self.shots_dataset[i][-1], hellinger_slice[-1], f'  {i}', verticalalignment='center')
        ax.set_ylim(0, 1)

    def plot_transition_points(self, circuit_index = 0, noise_index = 0, display_mean=True, tolerance=0.01):
        # Do not take into account noise levels beyond cutoff
        if noise_index >= self.cutoff_indices[circuit_index]:
            print(f"Warning: Noise index {noise_index} exceeds cutoff index {self.cutoff_indices[circuit_index]} for circuit {self.circuit_names[circuit_index]}. Results may be unreliable.")
            noise_index = self.cutoff_indices[circuit_index] - 1

        shots_slice = self.shots_dataset[circuit_index]
        hellinger_slice = self.hellinger[circuit_index][noise_index, :]
        hellinger_poly_slice = self.poly_hellinger[circuit_index][noise_index, :]
        mean_slice = self.mean_hamming[circuit_index][noise_index, :]
        std_slice = self.std_hamming[circuit_index][noise_index, :]
        std_poly_slice = self.poly_hamming_std[circuit_index][noise_index, :]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Display mean Hamming if requested
        if display_mean:
            ax.set_ylim(0, self.theoretical_max_Hamming + 0.1)
            ax.axhline(y=self.theoretical_max_Hamming, color='r', linestyle='--', label='Theoretical Max Hamming Distance')
            if noise_index != 0:
                ax.plot(shots_slice, mean_slice, color='b', label='hamming_mean')
        else:
            ax.set_ylim(0, 1)

        # Plot Hellinger distance and its fit
        A, B, _ = self.poly_hellinger_ABC[circuit_index][noise_index]
        ax.plot(shots_slice, hellinger_slice, color='c', label='hellinger')
        # hellinger_saturation_idx = np.where(np.abs(hellinger_poly_slice - C) < tolerance * A)[0]
        hellinger_saturation_idx = np.where(np.abs(-A * B * np.exp(-B * shots_slice)) < tolerance)[0]
        ax.plot(shots_slice, hellinger_poly_slice, color='r', label='hellinger_poly')
        ax.axvline(shots_slice[hellinger_saturation_idx[0]], color='r', linestyle='--', 
                label=f'Elbow at n={shots_slice[hellinger_saturation_idx[0]]:.0f}')

        # Plot Hamming std and its fit
        A, B, _ = self.poly_hamming_std_ABC[circuit_index][noise_index]
        if noise_index != 0:
            std_saturation_idx = np.where(np.abs(-A * B * np.exp(-B * shots_slice)) < tolerance)[0]
            ax.plot(shots_slice, std_slice, color='c', label='hamming_std')
            ax.plot(shots_slice, std_poly_slice, color='y', label='hamming_std_poly')
            ax.axvline(shots_slice[std_saturation_idx[0]], color='y', linestyle='--', 
                    label=f'Elbow at n={shots_slice[std_saturation_idx[0]]:.0f}')


    def hellinger_dashboard(self, noise_init=0):
        @interact
        def dashboard(
            noise_idx=FloatSlider(
                min=0,
                max=20,
                step=1,
                value=noise_init,
                description='Noise Intensity:',
                continuous_update=True
            )
        ):
            self.plot_hellinger(int(noise_idx))

    def transition_points_dashboard(self, circuit_init=0, noise_init=0, tolerance=0.01):
        @interact
        def dashboard(
            circuit_idx=Dropdown(
                options=[(name, idx) for idx, name in enumerate(self.circuit_names)],
                value=circuit_init,
                description='Circuit:',
            ),
            noise_idx=FloatSlider(
                min=0,
                max=20,
                step=1,
                value=noise_init,
                description='Noise Intensity:',
                continuous_update=True
            ),
            display_mean=Checkbox(
                value=True,
                description='Display Mean Hamming',
                disabled=False)
        ):
            self.plot_transition_points(int(circuit_idx), int(noise_idx), display_mean=display_mean, tolerance=tolerance)

def fit_exponential_decay_to_data(x_data, y_data):
    
    # Initial A, B, C guesses
    C_guess = np.mean(y_data[-5:])  # Estimate offset from last few points
    A_guess = np.max(y_data) - C_guess  # Amplitude from peak minus offset

    non_zero_mask = y_data - C_guess > 0.01
    if np.sum(non_zero_mask) > 2:
        log_y = np.log(y_data[non_zero_mask] - C_guess)
        x_for_fit = x_data[non_zero_mask]
        B_guess = -np.polyfit(x_for_fit, log_y, 1)[0]
    else:
        B_guess = 0.1  # Default guess

    initial_guess = [A_guess, B_guess, C_guess]

    # Apply Nonlinear Least Squares fitting for A, B, C
    try:
        # Basic fit
        params_opt, params_cov = curve_fit(
            exp_decay, 
            x_data, 
            y_data,
            p0=initial_guess,
            maxfev=10000  # Increase max function evaluations
        )
        
        A_fit, B_fit, C_fit = params_opt
        perr = np.sqrt(np.diag(params_cov))  # Parameter uncertainties
        
    except RuntimeError as e:
        print(f"Optimization failed: {e}")
        print("Trying with bounds...")
        
        # Try with bounds to help convergence
        params_opt, params_cov = curve_fit(
            exp_decay,
            x_data,
            y_data,
            p0=initial_guess,
            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),  # A,B ≥ 0
            maxfev=10000
        )
        
        A_fit, B_fit, C_fit = params_opt

    # return predicted values and parameters
    return A_fit, B_fit, C_fit

def exp_decay(x, A, B, C):
    return A * np.exp(-B * x) + C