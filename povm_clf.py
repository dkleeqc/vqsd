import time
import pennylane as qml
from pennylane import numpy as np
#dev = qml.device('qiskit.aer', wires=2)
#dev = qml.device('qiskit.ibmq', wires=2, shots=8192, ibmqx_token="e942e97ce86ca8c3609a4053fe6762ec3db17c41895b2d85d6a3560a1156501d57db36753f99138ecb32dca58a97ae5b38005ad39855dd92ab0e86cef852c1a2")



class State_Preparation():
    #qdev = qml.device("default.qubit", wires=2)
    def __init__(self, bloch_vec):
        self.bloch_vec = bloch_vec
        self.bloch_norm = np.dot(bloch_vec, bloch_vec)

        # Define Puali operators
        self.Paulis = np.zeros((3, 2, 2), dtype=complex)
        self.Paulis[0] = [[0, 1], [1, 0]]
        self.Paulis[1] = [[0, -1j], [1j, 0]]
        self.Paulis[2] = [[1, 0], [0, -1]]

        " Pure State (if) and Mixed State (else)"
        if np.isclose(self.bloch_norm, 1): 
            self._theta = np.arccos(self.bloch_vec[2])
            self._phi = np.arccos(self.bloch_vec[0]/np.sin(self._theta)) if not np.isclose(np.sin(self._theta), 0) else 0
            if self.bloch_vec[1] < 0:
                self._phi = (-1) * self._phi
            
            # Pure State vector
            self.statevector = [np.cos(self._theta/2), np.exp(1j*self._phi)*np.sin(self._theta/2)]

        else: 
            # Exact density matrix from a bloch vector
            self.density_matrix_from_bloch_vec = np.eye(2, dtype='complex128') / 2
            for i in range(3):
                self.density_matrix_from_bloch_vec += (self.bloch_vec[i] * self.Paulis[i]) / 2

            # Initial random parameters
            self._n_qubits = 2
            self._n_layers = 2
            self._prep_params = np.random.normal(0, np.pi, (self._n_qubits, self._n_layers, 3))
            
            # Run optimization of variational circuit.
            self.run_opt_4_prep()

            # Density matrix on circuit after optimization
            self.density_matrix_on_circ = self.density_matrix(self.output_bloch_v)

    
    def circuit(self, params, A=None):
        self._layer(params, [0, 1], 0)
        qml.CNOT(wires=[0,1])
        self._layer(params, [0, 1], 1)

        # returns the expectation of the input matrix A on the first qubit
        return qml.expval(qml.Hermitian(A, wires=0))


    def _layer(self, params, wires, j):
        for i in range(self._n_qubits):
            qml.Rot(params[i, j, 0], params[i, j, 1], params[i, j, 2], wires=wires[i])


    def cost_fn(self, params):
        cost = 0
        for k in range(3):
            qdev = qml.device("default.qubit", wires=2)
            qnode = qml.QNode(self.circuit, qdev)
            cost += np.abs(qnode(params, A=self.Paulis[k]) - self.bloch_vec[k])

        return cost


    def run_opt_4_prep(self, steps=300):
        # set up the optimizer
        opt = qml.AdamOptimizer()

        # the final stage of optimization isn't always the best, so we keep track of
        # the best parameters along the way
        best_cost = self.cost_fn(self._prep_params)
        self.best_prep_params = np.zeros((self._n_qubits, self._n_layers, 3))

        print("* Optimizing for preparing a mixed state", end=' ')
        #print("Cost after 0 steps is {:.4f}".format(self.cost_fn(self._prep_params)))

        # optimization begins
        for n in range(steps):
            self._prep_params = opt.step(self.cost_fn, self._prep_params)
            current_cost = self.cost_fn(self._prep_params)

            # keeps track of best parameters
            if current_cost < best_cost:
                self.best_prep_params = self._prep_params

            # Keep track of progress every 30 steps
            if n % 30 == 29 or n == steps - 1:
                #print("Cost after {} steps is {:.4f}".format(n + 1, current_cost))
                print("...",end='')
        print("")

        # calculate the Bloch vector of the output state
        self.output_bloch_v = np.zeros(3)
        for l in range(3):
            qdev = qml.device("default.qubit", wires=2)
            qnode = qml.QNode(self.circuit, qdev)
            self.output_bloch_v[l] = qnode(self.best_prep_params, A=self.Paulis[l])

        # print results
        print("Target Bloch vector = ", self.bloch_vec)
        print("Output Bloch vector = ", self.output_bloch_v)


    def state_prepared_on_circuit(self, wires):
        if np.isclose(self.bloch_norm, 1):
            qml.Rot(self._phi, self._theta, (-1)*self._phi, wires=wires)
        else:
            self._layer(self.best_prep_params, [wires[0],wires[1]], 0)
            qml.CNOT(wires=[wires[0],wires[1]])
            self._layer(self.best_prep_params, [wires[0],wires[1]], 1)


    def density_matrix(self, bloch_vector):

        rho = np.eye(2, dtype='complex128') / 2
        for i in range(3):
                rho += (bloch_vector[i] * self.Paulis[i]) / 2

        return rho




class SingleQubitPOVM(State_Preparation):
    def __init__(self, n=None, bloch_vec=None, *args, **kwargs):
        if bloch_vec != None:
            super().__init__(bloch_vec)
        self.n_outcome = n

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


    def three_element_povm(self, params, wires): 

        qml.PauliX(wires=wires[0])
        self.CCRY(params[0], wires=[wires[0], wires[1], wires[2]])
        qml.PauliX(wires=wires[0])
        self.CCRY(params[1], wires=[wires[0], wires[1], wires[2]])

        qml.PauliX(wires=wires[2])
        self.CCRot(params[2], params[3], params[4], wires=[wires[2],wires[1],wires[0]])
        qml.PauliX(wires=wires[2])
        self.CCRot(params[5], params[6], params[7], wires=[wires[2],wires[1],wires[0]])


    def CCRY(self, phi, wires):
        # 
        phi = phi/2
        qml.CRY(phi, wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CRY((-1)*phi, wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CRY(phi, wires=[wires[0], wires[2]])


    def CCRot(self, phi, theta, omega, wires):
        phi, theta, omega = phi/2, theta/2, omega/2
        qml.CRot(phi, theta, omega, wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CRot((-1)*omega, (-1)*theta, (-1)*phi, wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CRot(phi, theta, omega, wires=[wires[0], wires[2]])


    def povm_probs(self, params):
        
        if np.isclose(self.bloch_norm, 1):
            #self.initial_state(wires=self.wires[0])
            self.state_prepared_on_circuit(wires=self.wires[0])
        else:
            #self.initial_state(wires=[self.wires[0], self.wires[-1]])
            self.state_prepared_on_circuit(wires=[self.wires[0], self.wires[-1]])

        # arbitrary rotation
        self.first_U(params=params[:3], wire=self.wires[0])

        # two-element POVM module
        self.two_element_povm(params=params[3:], wires=[self.wires[0], self.wires[1]])
        
        if self.n_outcome == 3:
            self.three_element_povm(params[3 + 8:], wires=[self.wires[0], self.wires[1], self.wires[2]])
        
        
        if np.isclose(self.bloch_norm, 1):
            # For pure states
            return qml.probs(wires=self.wires[1:])
        else:
            # For mixed states
            return qml.probs(wires=self.wires[1:-1])


    # "Performing POVM on init_state(bloch_vec)"    
    def __call__(self, wires=None, dev=None):
        # wires
        self.wires = wires

        return qml.QNode(self.povm_probs, dev)


    """
    def run_expval(self):
        return qml.expval(qml.Identity(0) @ qml.PauliZ(1))
    """




class POVM_clf():
    def __init__(self, n, wires, bloch_vecs, devs):
        # number of outcomes
        self.n_outcome = n

        # initial random parameters
        np.random.seed(1)
        self.povm_params = 4 * np.pi * np.random.random([(3 + 8 * (n-1))])
        # Prior Probabilities for each state
        self.prior_probs = [1/self.n_outcome] * self.n_outcome
    
        
        self.density_matrices = []
        self.output_bloch_vecs = []
        self.output_density_matrices = []
        self.qnodes = qml.QNodeCollection()
        for i in range(self.n_outcome):
            # State Preparation as declaring an instance
            povm_circ = SingleQubitPOVM(n=self.n_outcome, bloch_vec=bloch_vecs[i])
            
            # Prepared density matrices
            self.density_matrices.append(povm_circ.density_matrix(bloch_vecs[i]))
            if not np.isclose(povm_circ.bloch_norm, 1):
                self.output_bloch_vecs.append(povm_circ.output_bloch_v)
                self.output_density_matrices.append(povm_circ.density_matrix(povm_circ.output_bloch_v))

            # Construct povm circuit whose output is the probability of a measurement outcome on ancilla qubits
            qnode = povm_circ(wires=wires, dev=devs[i]) # print(qnode(self.povm_params))
            self.qnodes.append(qnode)


    def cost_fn(self, x):
        probs_povm = self.qnodes(x, parallel=True)

        q = self.prior_probs

        if self.n_outcome == 2:
            res = 1 - (q[0] * probs_povm[0][0] + q[1] * probs_povm[1][1])
        if self.n_outcome == 3:
            res = 1 - (q[0] * probs_povm[0][0] + q[1] * probs_povm[1][2] + q[2] * probs_povm[2][3])
        return res


    def run_opt(self, steps=120):
        # initialize the optimizer
        opt = qml.GradientDescentOptimizer(stepsize=0.4)

        # cost fn for initial random params
        cost_list = [self.cost_fn(self.povm_params)]
        print("Cost(init_params)    : {: .7f}".format(cost_list[0]))

        # update the circuit parameters
        for i in range(steps):     # set the number of steps
            self.povm_params = opt.step(self.cost_fn, self.povm_params)
            cost_list.append(self.cost_fn(self.povm_params))
            if (i+1) % 20 == 0:
                print("Cost after step {:5d}: {: .7f}".format(i + 1, cost_list[i + 1]))

        #print("Optimized rotation angles: {}".format(self.povm_params))
        return cost_list


    def spectral_decomp(self, types='exact'):
        q = self.prior_probs
        rho = self.density_matrices if types == 'exact' else self.output_density_matrices

        lambdas, povm_bases = np.linalg.eig(q[0]*rho[0] - q[1]*rho[1])

        Helstrom_bound = (1 - np.sum(np.abs(lambdas))) / 2
        #optimal_povms = 

        return Helstrom_bound#, optimal_povms
        

    def unitaries(self):
        U = qml.Rot(self.povm_params[0], self.povm_params[1], self.povm_params[2], wires=100).matrix
        res = [U]
        for i in range(self.n_outcome-1):
            Ry0 = qml.RY(self.povm_params[3 + i*8], wires=100).matrix
            Ry1 = qml.RY(self.povm_params[4 + i*8], wires=100).matrix
            V0 = qml.Rot(self.povm_params[5 + i*8], self.povm_params[6 + i*8], self.povm_params[7 + i*8], wires=100).matrix
            V1 = qml.Rot(self.povm_params[8 + i*8], self.povm_params[9 + i*8], self.povm_params[10 + i*8], wires=100).matrix
            res += [Ry0, Ry1, V0, V1]

        return res


    def kraus_op(self):
        if self.n_outcome == 2:
            U, _, _, V0, V1 = self.unitaries()
            D0 = np.diag([np.cos(self.povm_params[3]/2), np.cos(self.povm_params[4]/2)])
            D1 = np.diag([np.sin(self.povm_params[3]/2), np.sin(self.povm_params[4]/2)])
            K0 = np.dot(np.dot(V0, D0), U)
            K1 = np.dot(np.dot(V1, D1), U)

            return K0, K1

        elif self.n_outcome == 3:
            U, _, _, V00, V01, _, _, V10, V11 = self.unitaries()
            D00 = np.diag([np.cos(self.povm_params[3]/2), np.cos(self.povm_params[4]/2)])
            D01 = np.diag([np.sin(self.povm_params[3]/2), np.sin(self.povm_params[4]/2)])
            D10 = np.diag([np.cos(self.povm_params[11]/2), np.cos(self.povm_params[12]/2)])
            D11 = np.diag([np.sin(self.povm_params[11]/2), np.sin(self.povm_params[12]/2)])
            K0 = np.dot(np.dot(V00, D00), U)
            _mid = np.dot(np.dot(V01, D01), U)
            K1 = np.dot(np.dot(V10,D10),_mid)
            K2 = np.dot(np.dot(V11,D11),_mid)

            return K0, K1, K2

    
    def povm(self):
        if self.n_outcome == 2:
            K0, K1 = self.kraus_op()
            E0 = np.dot(K0.conj().T, K0) 
            E1 = np.dot(K1.conj().T, K1)
            return E0, E1

        elif self.n_outcome == 3:
            K0, K1, K2 = self.kraus_op()
            E0 = np.dot(K0.conj().T, K0) 
            E1 = np.dot(K1.conj().T, K1)
            E2 = np.dot(K2.conj().T, K2)
            return E0, E1, E2



def state_2_bloch(state_vec):
    if state_vec[0] < 0:
        raise ValueError("The amplitude in state |0> doesn't have to be nonnegative.")
    theta = 2 * np.arccos(state_vec[0])
    #print(theta)

    phi = np.angle(state_vec[1]/np.sin(theta/2)) if not np.isclose(np.sin(theta/2), 0) else 0
    #print(phi)

    bloch_vec = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]

    return bloch_vec