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
    def __init__(self, names_dataset, shots_dataset, noise_dataset, hellinger, hamming_std,
                 hyperbolic_decay = None, exponential_decay = None, saturation_points = None):
        

        # --- Load dataset into the visualizer
        self.circuit_names = names_dataset
        self.shots_dataset = shots_dataset
        self.noise_dataset = noise_dataset
        self.hellinger = hellinger
        self.hamming_std = hamming_std

        if hyperbolic_decay is None and exponential_decay is None:
            raise ValueError("At least one of hyperbolic_decay or exponential_decay must be provided.")
        elif hyperbolic_decay is not None:
            self.poly_hellinger, self.poly_hamming_std, self.poly_predicted_hellinger = hyperbolic_decay
        else:
            self.poly_hellinger, self.poly_hamming_std, self.poly_predicted_hellinger = exponential_decay

        #--- Here we store the precomputed saturation points for all circuits and noise levels for specified slope range
        self.hellinger_saturation_points, self.predicted_hellinger_saturation_points = saturation_points
        
    def plot_graph(self, circuit_index = 0, noise_index = 0, slope_index = 0):

        shots_slice = self.shots_dataset[circuit_index]
        hellinger_slice = self.hellinger[circuit_index][noise_index, :]
        hamming_std_slice = self.hamming_std[circuit_index][noise_index, :]
    
        poly_hellinger_slice = self.poly_hellinger[circuit_index][noise_index, :]
        poly_hamming_std_slice = self.poly_hamming_std[circuit_index][noise_index, :]
        poly_predicted_hellinger_slice = self.poly_predicted_hellinger[circuit_index][noise_index, :]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_ylim(0, 1)

        #--- Plot Hamming std and its fit and saturation point
        ax.plot(shots_slice, hamming_std_slice, color='c', label='hamming_std_actual')
        ax.plot(shots_slice, poly_hamming_std_slice, color='y', label='hamming_std_poly')

        #--- Plot Hellinger distance and its fit and saturation point
        ax.plot(shots_slice, hellinger_slice, color='c', label='hellinger_actual')
        ax.plot(shots_slice, poly_hellinger_slice, color='r', label='hellinger_poly')
        ax.axvline(shots_slice[self.hellinger_saturation_points[circuit_index][noise_index][slope_index]], color='r', linestyle='--')

        # #--- Plot predicted Hellinger exponential decay and saturation point (based on the Hamming fit)
        ax.plot(shots_slice, poly_predicted_hellinger_slice, color='b', label='hellinger_predicted_poly')
        ax.axvline(shots_slice[self.predicted_hellinger_saturation_points[circuit_index][noise_index][slope_index]], color='b', linestyle='--')

        #--- Display legend and labels
        ax.set_xlabel('Number of Shots')
        ax.set_ylabel('Distance / Std')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    def plot_graph_dashboard(self, circuit_init=0, noise_init=0, slope_idx_init = 0):
       
        circuit_idx_widget = Dropdown(
            options=[(name, idx) for idx, name in enumerate(self.circuit_names)],
            value=circuit_init,
            description='Circuit:',
        )

        slope_threshold_widget = FloatSlider(
            min=0.00012,
            max=0.0005,
            step=0.00003,
            readout_format='.6f',
            value = slope_idx_init * 0.00003 + 0.00012,
            description='Slope Threshold:',
            continuous_update=True
        )

        @interact(
            circuit_idx=circuit_idx_widget,
            slope_threshold=slope_threshold_widget
        )
        def dashboard(circuit_idx, slope_threshold):
            self.plot_graph(int(circuit_idx), int(noise_init), slope_index=int((slope_threshold - 0.00012) / 0.00003))

        

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
    def plot_ABC_correlations(self):

        parameters = ['A', 'B', 'C']
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        for i, param in enumerate(parameters):
            ratios = []
            for circuit_idx in range(len(self.circuit_names)):

                hellinger_param = self.poly_hellinger_ABC[circuit_idx][0][i]
                hamming_param = self.poly_hamming_std_ABC[circuit_idx][0][i]
                if hamming_param != 0:
                    ratios.append(hellinger_param / hamming_param)

            axs[i].hist(ratios, bins=80, color='b', alpha=0.7, edgecolor='black')
            axs[i].set_title(f'Distribution of {param}_Hellinger / {param}_Hamming Ratios')
            axs[i].set_xlabel('Ratio Value')
            axs[i].set_ylabel('Frequency')
            axs[i].grid(True, alpha=0.3)

            # --- Fit a bell curve to the histogram and display mean and std on the plot
            mean = np.mean(ratios)
            std = np.std(ratios)
            x_fit = np.linspace(min(ratios), max(ratios), 100)
            y_fit = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fit - mean) / std) ** 2)
            axs[i].plot(x_fit, y_fit * len(ratios) * (max(ratios) - min(ratios)) / 80, color='r', label=f'Fit: μ={mean:.4f}, σ={std:.4f}')
            axs[i].legend()

        plt.tight_layout()
        plt.show()
