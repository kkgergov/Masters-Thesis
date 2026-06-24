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
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from ipywidgets import interact, FloatSlider, Checkbox, Dropdown

# Constants
SLOPE_THRESHOLD = 0.00015

class TransitionPointsVisualizer:
    def __init__(self, names_dataset, shots_dataset, noise_dataset, hamming_dataset, hellinger_dataset, n_qubits):
        
        #--- Here we store the precomputed saturation points for all circuits and noise levels for specified slope range
        self.poly_hellinger_ABC = None
        self.poly_hamming_std_ABC = None
        self.poly_hellinger = None
        self.poly_hamming_std = None

        self.hellinger_saturation_points = None
        self.hamming_saturation_points = None

        # --- For predicting Hellinger saturation points based on Hamming saturation points using general ratios
        self.predicted_hellinger_ABC = None
        self.predicted_hellinger = None

        self.predicted_hellinger_saturation_points = None

        # --- Load dataset into the visualizer
        self.circuit_names = names_dataset
        self.shots_dataset = shots_dataset
        self.noise_dataset = noise_dataset
        self.hamming_dataset = hamming_dataset
        self.hellinger = hellinger_dataset
        self.n_qubits = n_qubits

        self.theoretical_max_Hamming = n_qubits / 2

        #--- Precompute mean and std for each circuit in the Hamming dataset
        self.mean_hamming = [np.mean(hamming_data, axis=0) for hamming_data in hamming_dataset]
        self.std_hamming = [np.std(hamming_data, axis=0) for hamming_data in hamming_dataset]

        #--- Cutoff indices when Hamming exceeds 3.0
        self.cutoff_indices = np.array([len(noise_dataset[i]) for i in range(len(self.circuit_names))])
        for i in range(len(self.circuit_names)):
            for j in range(len(self.noise_dataset[i])):
                if self.mean_hamming[i][j, -1] >= 3.0:
                    self.cutoff_indices[i] = j
                    break

        #--- Precompute constants for Exponential Decay (24, 21, 3)
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
        # m_A, m_B, m_C = 1.63, 0.27, 12  # Tuned ratios
        m_A, m_B, m_C = 1.8007, 0.2767, 9.5682  # Ratios extracted from the mean of the data
        self.predicted_hellinger_ABC = np.array([
            [(m_A * self.poly_hamming_std_ABC[i][noise_idx][0], 
              m_B * self.poly_hamming_std_ABC[i][noise_idx][1], 
              m_C * self.poly_hamming_std_ABC[i][noise_idx][2]) 
             for noise_idx in range(self.hellinger[i].shape[0])]
            for i in range(len(self.circuit_names))
        ])

        #--- Precompute Exponential Decay (24, n_levels, n_shots)
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
        self.predicted_hellinger = np.array([
            [exp_decay(self.shots_dataset[i], *self.predicted_hellinger_ABC[i][noise_idx]) 
             for noise_idx in range(self.hellinger[i].shape[0])]
            for i in range(len(self.circuit_names))
        ])
        
    def plot_saturation_points(self, circuit_index = 0, noise_index = 0, slope_threshold = SLOPE_THRESHOLD, display_mean=True):
        #--- Don't display if Hamming > 3.0
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

        #--- Display mean Hamming if requested
        if display_mean:
            ax.plot(shots_slice, mean_slice, color='b', label='hamming_mean')
            ax.set_ylim(0, self.theoretical_max_Hamming + 0.1)
            ax.axhline(y=self.theoretical_max_Hamming, color='r', linestyle='--', label='Theoretical Max Hamming Distance')
        else:
            ax.set_ylim(0, 1)

        #--- Plot Hellinger distance and its fit and saturation point
        ax.plot(shots_slice, hellinger_slice, color='c', label='hellinger')
        ax.plot(shots_slice, hellinger_poly_slice, color='r', label='hellinger_poly')
        A, B, _ = self.poly_hellinger_ABC[circuit_index][noise_index]
        point_idx = np.where(np.abs(-A * B * np.exp(-B * self.shots_dataset[circuit_index])) < slope_threshold)[0]
        ax.axvline(shots_slice[point_idx[0]], color='r', linestyle='--')

        #--- Plot Hamming std and its fit and saturation point
        ax.plot(shots_slice, std_slice, color='c', label='hamming_std')
        ax.plot(shots_slice, std_poly_slice, color='y', label='hamming_std_poly')
        A, B, _ = self.poly_hamming_std_ABC[circuit_index][noise_index]
        point_idx = np.where(np.abs(-A * B * np.exp(-B * self.shots_dataset[circuit_index])) < slope_threshold)[0]
        ax.axvline(shots_slice[point_idx[0]], color='y', linestyle='--')

        #--- Plot predicted Hellinger exponential decay and saturation point (based on the Hamming fit)
        predicted_hellinger_slice = self.predicted_hellinger[circuit_index][noise_index, :]
        ax.plot(shots_slice, predicted_hellinger_slice, color='gray', label='predicted_hellinger_poly')
        A, B, _ = self.predicted_hellinger_ABC[circuit_index][noise_index]
        point_idx = np.where(np.abs(-A * B * np.exp(-B * self.shots_dataset[circuit_index])) < slope_threshold)[0]
    
        # shade a region around the predicted Hellinger saturation point
        ax.axvline(shots_slice[point_idx[0]], color='gray', linestyle='--')
        ax.fill_betweenx(
            y=[0, self.theoretical_max_Hamming + 0.1],
            x1=shots_slice[point_idx[0]] - 50,
            x2=shots_slice[point_idx[0]] + 50,
            color='gray',
            alpha=0.2,
            label='Predicted Region ±50 shots'
        )

        fig.canvas.draw_idle()

    def pre_compute_saturation_points(self):

        #--- Precompute saturation points for all circuits and noise levels for specified slope range
        self.hellinger_saturation_points = []
        self.hamming_saturation_points = []
        self.predicted_hellinger_saturation_points = []

        for circuit_idx in range(len(self.circuit_names)):
            hellinger_points_circuit = []
            hamming_points_circuit = []
            predicted_hellinger_points_circuit = []

            for noise_idx in range(self.noise_dataset[circuit_idx].shape[0]):
                hellinger_points_noise = []
                hamming_points_noise = []
                predicted_hellinger_points_noise = []

                for slope in np.arange(0.00012, 0.00120, 0.00003):
                    # Hellinger saturation point
                    A, B, _ = self.poly_hellinger_ABC[circuit_idx][noise_idx]
                    point_idx = np.where(np.abs(-A * B * np.exp(-B * self.shots_dataset[circuit_idx])) < slope)[0]
                    hellinger_points_noise.append((point_idx[0], self.shots_dataset[circuit_idx][point_idx[0]]))

                    # Hamming saturation point
                    A, B, _ = self.poly_hamming_std_ABC[circuit_idx][noise_idx]
                    point_idx = np.where(np.abs(-A * B * np.exp(-B * self.shots_dataset[circuit_idx])) < slope)[0]
                    hamming_points_noise.append((point_idx[0], self.shots_dataset[circuit_idx][point_idx[0]]))

                    # Predicted Hellinger saturation point
                    A, B, _ = self.predicted_hellinger_ABC[circuit_idx][noise_idx]
                    point_idx = np.where(np.abs(-A * B * np.exp(-B * self.shots_dataset[circuit_idx])) < slope)[0]
                    predicted_hellinger_points_noise.append((point_idx[0], self.shots_dataset[circuit_idx][point_idx[0]]))

                hellinger_points_circuit.append(hellinger_points_noise)
                hamming_points_circuit.append(hamming_points_noise)
                predicted_hellinger_points_circuit.append(predicted_hellinger_points_noise)

            self.hellinger_saturation_points.append(hellinger_points_circuit)
            self.hamming_saturation_points.append(hamming_points_circuit)
            self.predicted_hellinger_saturation_points.append(predicted_hellinger_points_circuit)

        self.hellinger_saturation_points = np.array(self.hellinger_saturation_points)
        self.hamming_saturation_points = np.array(self.hamming_saturation_points)
        self.predicted_hellinger_saturation_points = np.array(self.predicted_hellinger_saturation_points)

        #--- Precompute correlation between Hellinger and Hamming saturation points and fit linear regression
        self.saturation_correlation_params = []
        for circuit_idx in range(len(self.circuit_names)):
            circuit_params = []
            for noise_idx in range(self.noise_dataset[circuit_idx].shape[0]):
                hellinger_points = np.array(self.hellinger_saturation_points[circuit_idx][noise_idx])
                hamming_points = np.array(self.hamming_saturation_points[circuit_idx][noise_idx])

                # Fit linear regression
                m, b = np.polyfit(hellinger_points[:, 1], hamming_points[:, 1], 1)
                circuit_params.append((m, b))
            self.saturation_correlation_params.append(circuit_params)

        self.saturation_correlation_params = np.array(self.saturation_correlation_params)


    def plot_saturation_points_correlation(self, circuit_index = 0, noise_idx = 0):
        hellinger_points = np.array(self.hellinger_saturation_points[circuit_index][noise_idx])
        hamming_points = np.array(self.hamming_saturation_points[circuit_index][noise_idx])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)

        ax.scatter(hellinger_points[:, 1], hamming_points[:, 1], color='b')

        #--- Draw Linear Regression based on precomputed parameters
        x_fit = np.linspace(min(hellinger_points[:, 1]), max(hellinger_points[:, 1]), 100)
        m, b = self.saturation_correlation_params[circuit_index][noise_idx]
        y_fit = m * x_fit + b

        ax.plot(x_fit, y_fit, color='r', label=f'Fit: y = {m:.2f}x + {b:.2f}')
        ax.text(0.05, 0.95, f'Slope (m): {m:.4f}\nBias (b): {b:.2f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Hellinger Saturation Shot Count')
        ax.set_ylabel('Hamming Std Saturation Shot Count')
        ax.set_title(f'Saturation Points Correlation for Circuit: {self.circuit_names[circuit_index]} at Noise Index: {noise_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Plot a bell curve for A_Hellinger/ A_Hamming, B_Hellinger/ B_Hamming, C_Hellinger/ C_Hamming on 3 different plots
    def plot_ABC_correlations(self, circuit_indeces = [0], noise_index=None, display_bell_curve_fit=False, skip_circuit_noise_combinations=[]):

        parameters = ['A', 'B', 'C']
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        for i, param in enumerate(parameters):
            ratios = []
            for circuit_idx in circuit_indeces:

                noise_indices = range(self.cutoff_indices[circuit_idx])
                if noise_index is not None:
                    noise_indices = [noise_index]

                for noise_idx in noise_indices:
                    hellinger_param = self.poly_hellinger_ABC[circuit_idx][noise_idx][i]
                    hamming_param = self.poly_hamming_std_ABC[circuit_idx][noise_idx][i]
                    if hamming_param != 0 and (circuit_idx, noise_idx) not in skip_circuit_noise_combinations:
                        ratios.append(hellinger_param / hamming_param)

            axs[i].hist(ratios, bins=80, color='b', alpha=0.7, edgecolor='black')
            axs[i].set_title(f'Distribution of {param}_Hellinger / {param}_Hamming Ratios')
            axs[i].set_xlabel('Ratio Value')
            axs[i].set_ylabel('Frequency')
            axs[i].grid(True, alpha=0.3)

            # --- Fit a bell curve to the histogram and display mean and std
            if display_bell_curve_fit and len(ratios) > 1:
                mean = np.mean(ratios)
                std = np.std(ratios)
                x_fit = np.linspace(min(ratios), max(ratios), 100)
                y_fit = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fit - mean) / std) ** 2)
                axs[i].plot(x_fit, y_fit * len(ratios) * (max(ratios) - min(ratios)) / 80, color='r', label=f'Fit: μ={mean:.4f}, σ={std:.4f}')
                axs[i].legend()

        plt.tight_layout()
        plt.show()

    def transition_points_dashboard(self, circuit_init=0, noise_init=0, slope_threshold=SLOPE_THRESHOLD):

        circuit_idx_widget = Dropdown(
            options=[(name, idx) for idx, name in enumerate(self.circuit_names)],
            value=circuit_init,
            description='Circuit:',
        )
        noise_idx_widget = FloatSlider(
            min=0,
            max=self.cutoff_indices[circuit_init] - 1,
            step=1,
            value=noise_init,
            description='Noise Intensity:',
            continuous_update=True
        )
        slope_threshold_widget = FloatSlider(
            min=0.00012,
            max=0.00120,
            step=0.00003,
            readout_format='.6f',
            value=slope_threshold,
            description='Slope Threshold:',
            continuous_update=True
        )
        display_mean_widget = Checkbox(
            value=True,
            description='Display Mean Hamming',
            disabled=False)

        def update_slider_range(change):
            selected_circuit_idx = change['new']
            noise_idx_widget.max = self.cutoff_indices[selected_circuit_idx] - 1
            if noise_idx_widget.value > noise_idx_widget.max:
                noise_idx_widget.value = noise_idx_widget.max

        circuit_idx_widget.observe(update_slider_range, names='value')

        @interact(
            circuit_idx=circuit_idx_widget,
            noise_idx=noise_idx_widget,
            slope_threshold=slope_threshold_widget,
            display_mean=display_mean_widget
        )
        def dashboard(circuit_idx, noise_idx, slope_threshold, display_mean):
            self.plot_saturation_points(int(circuit_idx), int(noise_idx), slope_threshold=slope_threshold, display_mean=display_mean)

    def saturation_points_correlation_dashboard(self, circuit_init=0, noise_init=0):
        circuit_idx_widget=Dropdown(
            options=[(name, idx) for idx, name in enumerate(self.circuit_names)],
            value=circuit_init,
            description='Circuit:',
        )
        noise_idx_widget=FloatSlider(
            min=0,
            max=self.cutoff_indices[circuit_init] - 1,
            step=1,
            value=noise_init,
            description='Noise Intensity:',
            continuous_update=True
        )

        def update_slider_range(change):
            selected_circuit_idx = change['new']
            noise_idx_widget.max = self.cutoff_indices[selected_circuit_idx] - 1
            if noise_idx_widget.value > noise_idx_widget.max:
                noise_idx_widget.value = noise_idx_widget.max

        circuit_idx_widget.observe(update_slider_range, names='value')

        @interact(
            circuit_idx=circuit_idx_widget,
            noise_idx=noise_idx_widget
        )
        def dashboard(circuit_idx, noise_idx):
            self.plot_saturation_points_correlation(int(circuit_idx), int(noise_idx))
    
    def update_hamming_std_fit(self, new_ABC):
        self.poly_hamming_std_ABC = new_ABC

        self.poly_hamming_std = np.array([
            [exp_decay(self.shots_dataset[i], *self.poly_hamming_std_ABC[i][noise_idx]) 
             for noise_idx in range(self.std_hamming[i].shape[0])]
            for i in range(len(self.circuit_names))
        ])

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