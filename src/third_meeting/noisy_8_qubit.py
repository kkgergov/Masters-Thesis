import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.circuit.library import n_local
from qiskit.circuit.library import real_amplitudes
from qiskit.circuit.random import random_circuit

from utils import get_ideal_dist

def generate_initial_state():
    """
    Generate the initial state circuit and its corresponding bitstring.
    returns QuantumCircuit, str
    """

    # Initialize base circuit at |11011000>
    # This configuration ensures every possible activation for the [ry, rz] pairs

    qc = QuantumCircuit(8, 8)
    state_vector = np.zeros(2**8)
    state_vector[216] = 1  # 216 in binary is 11011000

    # Initialize the circuit with the state vector
    qc.initialize(state_vector, range(8))
    return qc, "11011000"

def generate_half_circuits():
    names = []
    halves = []

    # Circuit 0: QFT on 8 qubits
    names.append("QFT")
    half = QFTGate(8)
    halves.append(half)

    # Circuits 1-12: Variational circuits with different entanglements and reps
    entanglements = ['linear', 'circular', 'full']
    reps_list = [2, 4, 8, 20]

    for entanglement in entanglements:
        for reps in reps_list:
            names.append(f"varQC_{entanglement}_{reps}r")

            half = n_local(   
                num_qubits=8,
                rotation_blocks=['ry', 'rz'],
                entanglement_blocks='cx',
                entanglement=entanglement,
                reps=reps,
                skip_final_rotation_layer=True
            )
            param_list = list(half.parameters)
            random_values = np.random.uniform(0, 2*np.pi, size=len(param_list))
            half = half.assign_parameters(dict(zip(param_list, random_values)))
            halves.append(half)

    # Circuits 13-18: Real Amplitudes circuits with different entanglements and reps
    entanglements = ['linear', 'circular']
    reps_list = [2, 4, 8]

    for entanglement in entanglements:
        for reps in reps_list:
            names.append(f"RealAmp_{entanglement}_{reps}r")

            half = real_amplitudes(
                num_qubits=8,
                entanglement=entanglement,
                reps=reps
            )
            param_list = list(half.parameters)
            random_values = np.random.uniform(0, 2*np.pi, size=len(param_list))
            half = half.assign_parameters(dict(zip(param_list, random_values)))
            halves.append(half)

    # Circuits 19-23: Random circuits with varying depths
    depths = [5, 10, 20, 40, 80]
    for depth in depths:
        names.append(f"RandomCircuit_depth{depth}")

        half = random_circuit(
            num_qubits=8,
            depth=depth,
            max_operands=2,
            measure=False,
            conditional=False
        )
        halves.append(half)

    # Circuit 24: H gates on all qubits
    names.append("Hadamard")
    half = QuantumCircuit(8, 8)
    for qubit in range(8):
        half.h(qubit)
    halves.append(half)

    return names, halves

def generate_true_circuits(initial_state=None, halves=None):

    circuits = []
    true_dists = []

    initial_circuit, _ = initial_state

    for half in halves:
        qc = initial_circuit.compose(half)
        true_dists.append(get_ideal_dist(qc))
        qc.measure(range(8), range(8))
        circuits.append(qc)

    return circuits, true_dists

def generate_mirrored_circuits(initial_state=None, halves=None):
    
    circuits = []
    true_outputs = []

    initial_circuit, true_output = initial_state

    for half in halves:
        qc = initial_circuit.compose(half).compose(half.inverse())
        qc.measure(range(8), range(8))
        circuits.append(qc)
        true_outputs.append(true_output)

    return circuits, true_outputs

    """

    # Circuit 9: efficient SU2 circuit - circular entanglement 1 rep
    qc = QuantumCircuit(4, 4)
    esu2_circuit = EfficientSU2(
        num_qubits=4,
        entanglement='circular',
        reps=1
    )
    param_list = list(esu2_circuit.parameters)
    global random_values07
    random_values07 = np.random.uniform(0, 2*np.pi, size=len(param_list))
    esu2_circuit = esu2_circuit.assign_parameters(dict(zip(param_list, random_values07)))
    qc.append(esu2_circuit, [0, 1, 2, 3])
    qc.append(esu2_circuit.inverse(), [0, 1, 2, 3])
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)
    true_outputs.append("0000")

    # Circuit 10: efficient SU2 circuit - circular entanglement 4 reps
    qc = QuantumCircuit(4, 4)
    esu2_circuit = EfficientSU2(
        num_qubits=4,
        entanglement='circular',
        reps=4
    )
    param_list = list(esu2_circuit.parameters)
    global random_values08
    random_values08 = np.random.uniform(0, 2*np.pi, size=len(param_list))
    esu2_circuit = esu2_circuit.assign_parameters(dict(zip(param_list, random_values08)))
    qc.append(esu2_circuit, [0, 1, 2, 3])
    qc.append(esu2_circuit.inverse(), [0, 1, 2, 3])
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)
    true_outputs.append("0000")

    # Circuit 11: QFT on 8 qubits followed by inverse QFT on |10101010>
    qc = QuantumCircuit(8, 8)
    qc.x(1)
    qc.x(3)
    qc.x(5)
    qc.x(7)
    qft = QFTGate(8)
    qc.append(qft, range(8))
    qc.append(qft.inverse(), range(8))
    qc.measure(range(8), range(8))
    circuits.append(qc)
    true_outputs.append("10101010")    

    return circuits, true_outputs
    """

def get_circuits_and_outputs():
    initial_state = generate_initial_state()
    names, halves  = generate_half_circuits()
    true_circuits, true_dists = generate_true_circuits(initial_state, halves)
    mirrored_circuits, true_outputs = generate_mirrored_circuits(initial_state, halves)

    return {
        "names": names,
        "true_circuits": true_circuits,
        "true_dists": true_dists,
        "mirrored_circuits": mirrored_circuits,
        "true_outputs": true_outputs
    }