import pennylane as qml
from pennylane import numpy as np

#dev0 = qml.device('default.qubit', wires=2)
#dev1 = qml.device('default.qubit', wires=2)
#dev = qml.device('qiskit.aer', wires=2)
#dev = qml.device('qiskit.ibmq', wires=2, shots=8192, ibmqx_token="e942e97ce86ca8c3609a4053fe6762ec3db17c41895b2d85d6a3560a1156501d57db36753f99138ecb32dca58a97ae5b38005ad39855dd92ab0e86cef852c1a2")




class SingleQubitPOVM():
    def __init__(self, n=2, init_states=None, params=None, *args, **kwargs):
        self.num_outcome = n    # the number of outcomes
        self.init_states = init_states # (i x 2)-size array
        self.params = params if params 
        # *args and **kwarg for plug-in qml.dev


    def state_preparation(self, i):
        """
        State Preparation in quantum circuit 
        input: numpy.array of dimension (i x 2)
        i: the ith input state
        """
        theta = 2 * np.arccos(self.init_states[i][0])
        phi = np.angle(self.init_states[i][1]/np.sin(theta/2)) if not np.isclose(np.sin(theta/2), 0) else 0
        #print(theta, phi)

        qml.Rot(phi, theta, (-1)*phi, wires=0)

    
    def two_element_povm(self, params, wire0, wire1):
        "Two-Element POVM Module"
        

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



    def circuit_povm_expvalIZ(self, params, i):
        self.state_preparation(i)

        # arbitrary rotation
        qml.Rot(params[0], params[1], params[2], wires=0)

        # two-element POVM
        two_element_povm(3params[3:], 0, 1)

        return qml.expval(qml.Identity(0) @ qml.PauliZ(1))


    def __call__(self, **kwarg):
        dev0 = qml.device('default.qubit', wires=2)
        dev1 = qml.device('default.qubit', wires=2)
        qnodes = qml.QNodeCollection(
            [qml.QNode(circuit_povm_expvalIZ(i=0), dev0, interface="tf"), 
            qml.QNode(circuit_povm_expvalIZ(i=1), dev1, interface="tf")])
        
        qnodes

        https://pennylane.readthedocs.io/en/stable/code/api/pennylane.QNodeCollection.html
        return 


    def unitaries_in_povm_mo(params):
        U = qml.Rot(params[0], params[1], params[2], wires=2).matrix
        Ry0 = qml.RY(params[3], wires=2).matrix
        Ry1 = qml.RY(params[4], wires=2).matrix
        V0 = qml.Rot(params[5], params[6], params[7], wires=2).matrix
        V1 = qml.Rot(params[8], params[9], params[10], wires=2).matrix

        return U, Ry0, Ry1, V0, V1


    def kraus_op(params):
        U, _, _, V0, V1 = unitaries_in_povm(params)
        D0 = np.diag([np.cos(params[3]/2), np.cos(params[4]/2)])
        D1 = np.diag([np.sin(params[3]/2), np.sin(params[4]/2)])
        K0 = np.dot(np.dot(V0, D0), U)
        K1 = np.dot(np.dot(V1, D1), U)

        return K0, K1



class POVM_clf(SingleQubitPOVM):
    def __init__(self, n=2, init_state=None):
        super().__init__(n=n, init_state=init_state)



    def cost(self, x):
        K0psi0 = circuit_povm_expvalIZpsi0(x)
        K1psi1 = circuit_povm_expvalIZpsi1(x)
        return (1/2) * (1 - (1/2)*K0psi0 + (1/2)*K1psi1)




