import pennylane as qml
from pennylane import numpy as np

#dev = qml.device('qiskit.aer', wires=2)
#dev = qml.device('qiskit.ibmq', wires=2, shots=8192, ibmqx_token="e942e97ce86ca8c3609a4053fe6762ec3db17c41895b2d85d6a3560a1156501d57db36753f99138ecb32dca58a97ae5b38005ad39855dd92ab0e86cef852c1a2")


class SingleQubitPOVM():
    def __init__(self, n=2, *args, **kwargs):
        self.num_outcome = n    # the number of outcomes, only possible for n=2, 3
        
        
    def state_preparation(self, state=[1,0], wire=0, types='pure'):
        """
        State Preparation in quantum circuit 
        input: list of length 2
        i: the ith input state
        """
        if types=='pure': 
            theta = 2 * np.arccos(state[0])
            phi = np.angle(state[1]/np.sin(theta/2)) if not np.isclose(np.sin(theta/2), 0) else 0
            #print(theta, phi)

            qml.Rot(phi, theta, (-1)*phi, wires=wire)


    def first_U(self, params, wire):
        qml.Rot(params[0], params[1], params[2], wires=wire)


    def two_element_povm(self, params, wires):

        # Controlled-RY gate controlled by first qubit in |0> state
        qml.PauliX(wires=wires[0])
        qml.CRY(params[0], wires=[wires[0],wires[1]])
        qml.PauliX(wires=wires[0])
        
        # Controlled-RY gate controlled by first qubit in |1> state
        qml.CRY(params[1], wires=[wires[0],wires[1]])
        
        # Controlled-Rotation gate (arbitrary single-qubit unitary operator) controlled by 2nd qubit in |0> state
        qml.PauliX(wires=wires[1])
        qml.CRot(params[2], params[3], params[4], wires=[wires[1],wires[0]])
        qml.PauliX(wires=wires[1])

        # # Controlled-Rotation gate (arbitrary single-qubit unitary operator) controlled by 2nd qubit in |1> state
        qml.CRot(params[5], params[6], params[7], wires=[wires[1],wires[0]])


    def three_eleemnt_povm(self, params, wires):
        return

    # "Performing POVM on init_state"    
    def __call__(self, init_state=None, wires=None, dev=None):
        # wires
        self.init_state = init_state
        self.wires = wires

        return qml.QNode(self.povm_probs, dev)
        


    def povm_probs(self, params):
        self.state_preparation(state=self.init_state, wire=self.wires[0]) # initial state, type: list

        # arbitrary rotation
        self.first_U(params[:3], self.wires[0])

        # two-element POVM module
        self.two_element_povm(params[3:], [self.wires[0],self.wires[1]])

        return qml.probs(wires=self.wires[1])


    """
    def run_expval(self):
        return qml.expval(qml.Identity(0) @ qml.PauliZ(1))
    """




    def unitaries_in_povm_mo(self, params):
        U = qml.Rot(params[0], params[1], params[2], wires=2).matrix
        Ry0 = qml.RY(params[3], wires=2).matrix
        Ry1 = qml.RY(params[4], wires=2).matrix
        V0 = qml.Rot(params[5], params[6], params[7], wires=2).matrix
        V1 = qml.Rot(params[8], params[9], params[10], wires=2).matrix

        return U, Ry0, Ry1, V0, V1


    def kraus_op(self, params):
        U, _, _, V0, V1 = unitaries_in_povm(params)
        D0 = np.diag([np.cos(params[3]/2), np.cos(params[4]/2)])
        D1 = np.diag([np.sin(params[3]/2), np.sin(params[4]/2)])
        K0 = np.dot(np.dot(V0, D0), U)
        K1 = np.dot(np.dot(V1, D1), U)

        return K0, K1




class POVM_clf(SingleQubitPOVM):
    def __init__(self, n, wires, init_states, dev):
        
        # initial random parameters
        self.params = 2 * np.pi * np.random.random([(3 + 8 * 2)])
        
        self.qnodes = qml.QNodeCollection()
        for i in range(2):
            povm_circ = SingleQubitPOVM()
            qnode = povm_circ(init_state=init_states[i], wires=wires, dev=dev[i])
            self.qnodes.append(qnode)


    def cost_fn(self, x):
        probs_povm = self.qnodes(x, parallel=True)

        q = [1/2, 1/2]

        res = 1 - (q[0] * probs_povm[0][0] + q[1] * probs_povm[1][1])
        return res


    def run_opt(self, steps=120):

        # initialize the optimizer
        opt = qml.GradientDescentOptimizer(stepsize=0.4)

        # cost fn 
        print("Cost(init_params) =", self.cost_fn(self.params))


        # update the circuit parameters
        cost_list = [self.cost_fn(self.params)]
        for i in range(steps):     # set the number of steps
            self.params = opt.step(self.cost_fn, self.params)
            cost_list.append(self.cost_fn(self.params))
            if (i+1) % 5 == 0:
                print("Cost after step {:5d}: {: .7f}".format(i + 1, cost_list[i + 1]))

        #print("Optimized rotation angles: {}".format(self.params))
        return cost_list

