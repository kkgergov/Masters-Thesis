import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

def create_noise_model(gate_error=0.01, measurement_error=0.01):
    """Create a noise model with depolarizing and measurement errors"""
    noise_model = NoiseModel()
    
    # Add depolarizing error to single-qubit gates
    single_qubit_error = depolarizing_error(gate_error, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'x', 'y', 'z'])
    
    # Add depolarizing error to two-qubit gates
    two_qubit_error = depolarizing_error(gate_error, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
    
    # Add measurement error
    measurement_error_model = depolarizing_error(measurement_error, 1)
    noise_model.add_all_qubit_quantum_error(measurement_error_model, 'measure')
    
    return noise_model