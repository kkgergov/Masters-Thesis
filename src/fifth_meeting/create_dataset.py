import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from third_meeting.noisy_8_qubit import get_circuits_and_outputs
from utils import TransitionPointsVisualizer
import numpy as np
import supermarq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create training dataset with input features: communication, entanglement, liveness, parallelism
# and output labels: Hamming decay A, B, C parameters

def get_supermarq_features(circuits=None):

    features = []
    for circuit in circuits:
        comm = supermarq.features.compute_communication_with_qiskit(circuit)
        entanglement = supermarq.features.compute_entanglement_with_qiskit(circuit)
        liveness = supermarq.features.compute_liveness_with_qiskit(circuit)
        parallelism = supermarq.features.compute_parallelism_with_qiskit(circuit)

        features.append([comm, entanglement, liveness, parallelism])

    return np.array(features)

def get_noise_levels():

    loaded = np.load(f'data/third_meeting/all_circuits_data.npz', allow_pickle=True)
    noise_dataset = loaded['noise_dataset']

    return noise_dataset

# ----------------------------------------------------------------
# 1. MODEL ARCHITECTURE
# ----------------------------------------------------------------
class HammingDecayPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HammingDecayPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # Direct output to 3 nodes
        )

    def forward(self, x):
        # Outputs scaled predictions centered around 0
        return self.network(x)

if __name__ == "__main__":
    print(os.getcwd())
    get_noise_levels()
