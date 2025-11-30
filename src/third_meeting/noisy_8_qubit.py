import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.circuit.library import n_local
from qiskit.circuit.library import EfficientSU2

from utils import get_ideal_dist

random_values00 = []
random_values01 = []
random_values02 = []
random_values03 = []
random_values04 = []
random_values05 = []
random_values06 = []
random_values07 = []
random_values08 = []
random_values09 = []

def generate_first_half_circuits():
    halves = []

    # Circuit 0: QFT on 8 qubits
    half = QuantumCircuit(8, 8)
    qft = QFTGate(8)
    half.append(qft, range(8))
    half.measure(range(8), range(8))
    halves.append(half)

def generate_circuits():

    circuits = []
    true_dists = []


    # Circuit 7: variational circuit 5 - full entanglement 12 reps
    qc = QuantumCircuit(4, 4)
    var_qc = n_local(   
        num_qubits=4,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='full',
        reps=12,
        skip_final_rotation_layer=True
    )
    param_list = list(var_qc.parameters)
    var_qc = var_qc.assign_parameters(dict(zip(param_list, random_values05)))
    qc.append(var_qc, [0, 1, 2, 3])
    # Finish up circuit and calculate theoretical perfect output dist
    true_dists.append(get_ideal_dist(qc))
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)

    # Circuit 8: variational circuit 6 - circular entanglement 12 reps
    qc = QuantumCircuit(4, 4)
    var_qc = n_local(   
        num_qubits=4,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='circular',
        reps=12,
        skip_final_rotation_layer=True
    )
    param_list = list(var_qc.parameters)
    var_qc = var_qc.assign_parameters(dict(zip(param_list, random_values06)))
    qc.append(var_qc, [0, 1, 2, 3])
    # Finish up circuit and calculate theoretical perfect output dist
    true_dists.append(get_ideal_dist(qc))
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)

    # Circuit 9: efficient SU2 circuit - circular entanglement 1 rep
    qc = QuantumCircuit(4, 4)
    var_qc = EfficientSU2(
        num_qubits=4,
        entanglement='circular',
        reps=1
    )
    param_list = list(var_qc.parameters)
    var_qc = var_qc.assign_parameters(dict(zip(param_list, random_values07)))
    qc.append(var_qc, [0, 1, 2, 3])
    # Finish up circuit and calculate theoretical perfect output dist
    true_dists.append(get_ideal_dist(qc))
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)

    # Circuit 10: efficient SU2 circuit - circular entanglement 4 reps
    qc = QuantumCircuit(4, 4)
    var_qc = EfficientSU2(
        num_qubits=4,
        entanglement='circular',
        reps=4
    )
    param_list = list(var_qc.parameters)
    var_qc = var_qc.assign_parameters(dict(zip(param_list, random_values08)))
    qc.append(var_qc, [0, 1, 2, 3])
    # Finish up circuit and calculate theoretical perfect output dist
    true_dists.append(get_ideal_dist(qc))
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)

    # Circuit 11: QFT on 8 qubits on |10101010>
    qc = QuantumCircuit(8, 8)
    qc.x(1)
    qc.x(3)
    qc.x(5)
    qc.x(7)
    qft = QFTGate(8)
    qc.append(qft, range(8))
    # Finish up circuit and calculate theoretical perfect output dist
    true_dists.append(get_ideal_dist(qc))
    qc.measure(range(8), range(8))
    circuits.append(qc)

    return circuits, true_dists

def generate_mirrored_circuits():
    
    circuits = []
    true_outputs = []

    """Circuit 0: Three qubits with two H gates"""
    qc = QuantumCircuit(3, 3)
    for _ in range(4):
        qc.h(0)
    for _ in range(4):
        qc.h(1)
    for _ in range(4):
        qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc)
    true_outputs.append("000")

    """Circuit 1: 3 qubit QFT followed by inverse QFT on |101>"""
    qc = QuantumCircuit(3, 3)
    qc.initialize([0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 2])
    qft = QFTGate(3)
    qc.append(qft, [0, 1, 2])
    qc.append(qft.inverse(), [0, 1, 2])
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc)
    true_outputs.append("101")

    """Circuit 2: variational circuit 0"""
    qc = QuantumCircuit(3, 3)
    qc.initialize([0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 2])
    # Prepare and append mirrored Ansatz
    var_qc = n_local(3, "ry", "cx", "linear", reps=2, insert_barriers=True)
    param_list = list(var_qc.parameters)
    global random_values00
    random_values00 = np.random.uniform(0, 2*np.pi, size=len(param_list))
    var_qc = var_qc.assign_parameters(dict(zip(param_list, random_values00)))
    qc.append(var_qc, [0, 1, 2])
    qc.append(var_qc.inverse(), [0, 1, 2])

    # Finish up circuit
    qc.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc)
    true_outputs.append("101")

    # 4 qubit circuits from now on
    """Circuit 3: variational circuit 1 - Linear entanglement 2 reps"""
    qc = QuantumCircuit(4, 4)

    # Linear entanglement
    linear_circuit = n_local(
        num_qubits=4,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='linear',
        reps=2,
        skip_final_rotation_layer=True
    )
    param_list = list(linear_circuit.parameters)
    global random_values01
    random_values01 = np.random.uniform(0, 2*np.pi, size=len(param_list))
    linear_circuit = linear_circuit.assign_parameters(dict(zip(param_list, random_values01)))
    qc.append(linear_circuit, [0, 1, 2, 3])
    qc.append(linear_circuit.inverse(), [0, 1, 2, 3])

    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)
    true_outputs.append("0000")

    """Circuit 4: variational circuit 2 - circular entanglement 2 reps"""
    qc = QuantumCircuit(4, 4)
    circular_circuit = n_local(
        num_qubits=4,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='circular',
        reps=2,
        skip_final_rotation_layer=True
    )
    param_list = list(circular_circuit.parameters)
    global random_values02
    random_values02 = np.random.uniform(0, 2*np.pi, size=len(param_list))
    circular_circuit = circular_circuit.assign_parameters(dict(zip(param_list, random_values02)))
    qc.append(circular_circuit, [0, 1, 2, 3])
    qc.append(circular_circuit.inverse(), [0, 1, 2, 3])
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)
    true_outputs.append("0000")

    ## Circuit 5: variational circuit 3 - full entanglement 2 reps
    qc = QuantumCircuit(4, 4)
    full_circuit = n_local(
        num_qubits=4,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='full',
        reps=2,
        skip_final_rotation_layer=True
    )
    param_list = list(full_circuit.parameters)
    global random_values03
    random_values03 = np.random.uniform(0, 2*np.pi, size=len(param_list))
    full_circuit = full_circuit.assign_parameters(dict(zip(param_list, random_values03)))
    qc.append(full_circuit, [0, 1, 2, 3])
    qc.append(full_circuit.inverse(), [0, 1, 2, 3])
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)
    true_outputs.append("0000")

    # Circuit 6: variational circuit 4 - full entanglement 4 reps
    qc = QuantumCircuit(4, 4)
    full_circuit = n_local(
        num_qubits=4,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='full',
        reps=4,
        skip_final_rotation_layer=True
    )
    param_list = list(full_circuit.parameters)
    global random_values04
    random_values04 = np.random.uniform(0, 2*np.pi, size=len(param_list))
    full_circuit = full_circuit.assign_parameters(dict(zip(param_list, random_values04)))
    qc.append(full_circuit, [0, 1, 2, 3])
    qc.append(full_circuit.inverse(), [0, 1, 2, 3])
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)
    true_outputs.append("0000")

    # Circuit 7: variational circuit 5 - full entanglement 12 reps
    qc = QuantumCircuit(4, 4)
    full_circuit = n_local(
        num_qubits=4,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='full',
        reps=12,
        skip_final_rotation_layer=True
    )
    param_list = list(full_circuit.parameters)
    global random_values05
    random_values05 = np.random.uniform(0, 2*np.pi, size=len(param_list))
    full_circuit = full_circuit.assign_parameters(dict(zip(param_list, random_values05)))
    qc.append(full_circuit, [0, 1, 2, 3])
    qc.append(full_circuit.inverse(), [0, 1, 2, 3])
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)
    true_outputs.append("0000")

    # Circuit 8: variational circuit 6 - circular entanglement 12 reps
    qc = QuantumCircuit(4, 4)
    var_qc = n_local(
        num_qubits=4,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='circular',
        reps=12,
        skip_final_rotation_layer=True
    )
    param_list = list(var_qc.parameters)
    global random_values06
    random_values06 = np.random.uniform(0, 2*np.pi, size=len(param_list))
    var_qc = var_qc.assign_parameters(dict(zip(param_list, random_values06)))
    qc.append(var_qc, [0, 1, 2, 3])
    qc.append(var_qc.inverse(), [0, 1, 2, 3])
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    circuits.append(qc)
    true_outputs.append("0000")

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