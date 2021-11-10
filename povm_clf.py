import pennylane as qml
from pennylane import numpy as np

dev = qml.device('default.qubit', wires=2)
#dev = qml.device('qiskit.aer', wires=2)
#dev = qml.device('qiskit.ibmq', wires=2, shots=8192, ibmqx_token="e942e97ce86ca8c3609a4053fe6762ec3db17c41895b2d85d6a3560a1156501d57db36753f99138ecb32dca58a97ae5b38005ad39855dd92ab0e86cef852c1a2")


def two_element_povm(params, wire0, wire1):
    #params 

    # Controlled-RY gate controlled by first qubit in |0> state
    qml.PauliX(wires=wire0)
    qml.CRY(params[0], wires=[wire0,wire1])
    qml.PauliX(wires=wire0)
    
    # Controlled-RY gate controlled by first qubit in |1> state
    qml.CRY(params[1], wires=[wire0,wire1])

    # Controlled-Rotation gate (arbitrary single-qubit unitary operator) controlled by 2nd qubit in |0> state
    qml.PauliX(wires=wire1)
    qml.CRot(params[2], params[3], params[4], wires=[wire1,wire0])
    qml.PauliX(wires=wire1)

    # # Controlled-Rotation gate (arbitrary single-qubit unitary operator) controlled by 2nd qubit in |1> state
    qml.CRot(params[5], params[6], params[7], wires=[wire1,wire0])