import time
import inspect
import pennylane as qml
from pennylane import numpy as np

from scipy.linalg import sqrtm, expm

import pandas as pd
import cvxpy as cp

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
#dev = qml.device('qiskit.aer', wires=2)
#dev = qml.device('qiskit.ibmq', wires=2, shots=8192, ibmqx_token="e942e97ce86ca8c3609a4053fe6762ec3db17c41895b2d85d6a3560a1156501d57db36753f99138ecb32dca58a97ae5b38005ad39855dd92ab0e86cef852c1a2")
from IPython.display import display #'display' is not defined so I introduced it here but I'll check this later! 


class State_Preparation():
    #qdev = qml.device("default.qubit", wires=2)
    def __init__(self, bloch_vec):
        self.bloch_vec = bloch_vec
        self.purity_check = np.isclose(np.linalg.norm(bloch_vec),1)

        " Pure State (if) and Mixed State (else)"
        if self.purity_check:
            self._theta = np.arccos(self.bloch_vec[2])
            self._phi = np.arccos(self.bloch_vec[0]/np.sin(self._theta)) if not np.isclose(np.sin(self._theta), 0) else 0
            if self.bloch_vec[1] < 0:
                self._phi = (-1) * self._phi
            
            # Pure State vector
            self.statevector = [np.cos(self._theta/2), np.exp(1j*self._phi)*np.sin(self._theta/2)]

        else: 
            # Exact density matrix from a bloch vector
            self.density_matrix = bloch_2_state(bloch_vec, 'density')

            # Initial random parameters
            self._n_qubits = 2
            self._n_layers = 2
            self._prep_params = np.random.normal(0, 2 * np.pi, (self._n_qubits, self._n_layers, 3))
            
            # Run optimization of variational circuit.
            self.run_opt_4_prep()

            # Density matrix on circuit after optimization
            self.density_matrix_on_circ = bloch_2_state(self.output_bloch_v, 'density')

    
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
        # Define Puali operators
        self.Paulis = np.zeros((3, 2, 2), dtype=complex)
        self.Paulis[0] = [[0, 1], [1, 0]]
        self.Paulis[1] = [[0, -1j], [1j, 0]]
        self.Paulis[2] = [[1, 0], [0, -1]]

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


    def on_circuit(self, wires):
        if self.purity_check:
            qml.Rot(self._phi, self._theta, (-1)*self._phi, wires=wires)
        else:
            self._layer(self.best_prep_params, [wires[0],wires[1]], 0)
            qml.CNOT(wires=[wires[0],wires[1]])
            self._layer(self.best_prep_params, [wires[0],wires[1]], 1)






def rdm(state):
    res = state.reshape(2,2)
    return np.dot(res, res.conjugate().T)


def rho_2_bloch(rho):
    bloch_vec = []
    bloch_vec.append(2*np.real(rho[0,1]))
    bloch_vec.append(2*np.imag(rho[1,0]))
    bloch_vec.append(np.real(rho[0,0]-rho[1,1]))
    
    return np.array(bloch_vec)


def state_2_bloch(state_vec):
    if state_vec[0] < 0:
        for i in range(len(state_vec)):
            state_vec[i] = (-1) * state_vec[i]
    theta = 2 * np.arccos(state_vec[0])
    #print(theta)

    phi = np.angle(state_vec[1]/np.sin(theta/2)) if not np.isclose(np.sin(theta/2), 0) else 0
    #print(phi)

    bloch_vec = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]

    return bloch_vec


def bloch_2_state(bloch_vec=None, output_type='density'):
    if len(bloch_vec) != 3:
        raise ValueError("The length 'bloch_vec' must be 3.")
    
    " Pure State (if) and Mixed State (else)"
    bloch_norm = np.abs(np.dot(bloch_vec, bloch_vec))
    if np.isclose(bloch_norm, 1) and output_type == 'vector': 
        _theta = np.arccos(bloch_vec[2])
        _phi = np.arccos(bloch_vec[0]/np.sin(_theta)) if not np.isclose(np.sin(_theta), 0) else 0
        if bloch_vec[1] < 0:
            _phi = (-1) * _phi
        
        # Pure State vector
        rho = [np.cos(_theta/2), np.exp(1j*_phi)*np.sin(_theta/2)]

    else:
        Paulis = np.zeros((3, 2, 2), dtype=complex)
        Paulis[0] = [[0, 1], [1, 0]]
        Paulis[1] = [[0, -1j], [1j, 0]]
        Paulis[2] = [[1, 0], [0, -1]]

        rho = np.eye(2, dtype='complex128') / 2
        for i in range(3):
            rho += (bloch_vec[i] * Paulis[i]) / 2

    return rho





class POVM_Clf():
    def __init__(self, n, wires, devs, a_priori_probs, input_states, state_wires):
        # number of outcomes
        self.n_outcome = n

        # number of ancillary qubits
        num_ancilla_qubits = int(np.ceil(np.log(n) / np.log(2)))
        num_measuring_qubits = int(len(wires) - num_ancilla_qubits)

        # Prior Probabilities for each state
        self.a_priori_probs = a_priori_probs

        # initial random parameters
        np.random.seed(1)
        num_params = (2**(2*num_measuring_qubits)-1+2**num_measuring_qubits) * (n-1)
        self.povm_params = 2 * np.pi * np.random.random([num_params])
        


        self.qnodes = qml.QNodeCollection()
        for i in range(self.n_outcome):            
            if num_measuring_qubits == 1:
                povm_circ = SingleQubitPOVM(wires=wires, dev=devs[i])
            else:
                povm_circ = TwoQubitPOVM(wires=wires, dev=devs[i])
            
            # Construct povm circuit whose output is the probability of a measurement outcome on ancilla qubits
            qnode = povm_circ(n, input_states[i], state_wires)
            self.qnodes.append(qnode)
            

    def cost_fn(self, x):
        probs_povm = self.qnodes(x)
        q = self.a_priori_probs

        res = 1
        for i in range(self.n_outcome):
            res -= q[i] * probs_povm[i][i]
        return res


    def run_opt(self, steps=1000):
        # initialize the optimizer
        #opt = qml.GradientDescentOptimizer(stepsize=0.4)
        opt = qml.AdamOptimizer(stepsize=0.02)

        # cost fn for initial random params
        cost_list = [self.cost_fn(self.povm_params)]
        print("Cost(init_params)    : {:.7f}".format(cost_list[0]))

        # update the circuit parameters
        for i in range(steps):     # set the number of steps
            self.povm_params = opt.step(self.cost_fn, self.povm_params)
            cost_list.append(self.cost_fn(self.povm_params))
            if (i+1) % 20 == 0:
                print("Cost after step {:5d}: {: .7f}".format(i + 1, cost_list[i + 1]))

            if i>1 and cost_list[i-1] - cost_list[i] < 1e-7:
                break
        #print("Optimized rotation angles: {}".format(self.povm_params))
        return cost_list
        

    def unitaries(self):
        res = []
        for i in range(self.n_outcome-1):
            U = qml.Rot(self.povm_params[0 + i*5], self.povm_params[1 + i*5], self.povm_params[2 + i*5], wires=100).matrix
            Ry0 = qml.RY(self.povm_params[3 + i*5], wires=100).matrix
            Ry1 = qml.RY(self.povm_params[4 + i*5], wires=100).matrix

            res += [U, Ry0, Ry1]

        return res

    # Kraus Operator
    def kraus_op(self):
        if self.n_outcome == 2:
            rots = self.unitaries()
            U = rots[0]
            D0 = np.diag([np.cos(self.povm_params[3]/2), np.cos(self.povm_params[4]/2)])
            D1 = np.diag([np.sin(self.povm_params[3]/2), np.sin(self.povm_params[4]/2)])
            K0 = np.dot(D0, U)
            K1 = np.dot(D1, U)

            return K0, K1

        elif self.n_outcome == 3:
            U0, _, _, U1, _, _ = self.unitaries()
            D00 = np.diag([np.cos(self.povm_params[3]/2), np.cos(self.povm_params[4]/2)])
            D01 = np.diag([np.sin(self.povm_params[3]/2), np.sin(self.povm_params[4]/2)])
            D10 = np.diag([np.cos(self.povm_params[8]/2), np.cos(self.povm_params[9]/2)])
            D11 = np.diag([np.sin(self.povm_params[8]/2), np.sin(self.povm_params[9]/2)])
            K0 = np.dot(D00, U0)
            _mid = np.dot(D01, U0)
            K1 = np.dot(np.dot(D10, U1),K0)
            K2 = np.dot(np.dot(D11, U1),K0)

            return K0, K1, K2

    # Poisitive Operator Valued Measurement
    def povm_matrices(self):
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


    def dp_res(self):
        rho = self.density_matrices
        rho_pd = [pd.DataFrame(rho[i], columns=['rho_i', ''], index=['i='+str(i),'']) for i in range(self.n_outcome)]
        rho_pd = pd.concat(rho_pd)

        E = self.povm()
        E_pd = [pd.DataFrame(E[i], columns=['E_i', ''], index=['i='+str(i),'']) for i in range(self.n_outcome)]
        E_pd = pd.concat(E_pd)
        tr_E_pd = [pd.DataFrame(['',np.real(np.trace(np.dot(rho[i], E[i])))], columns=['Tr[rho_i.E_i]'], index=['i='+str(i),'']) for i in range(self.n_outcome)]
        tr_E_pd = pd.concat(tr_E_pd)

        #K = self.kraus_op()
        #K_pd = [pd.DataFrame(K[i], columns=['K_i', ''], index=['i='+str(i),'']) for i in range(3)]
        #K_pd = pd.concat(K_pd)

        pgm, _ = self.pgm()
        pgm_pd = [pd.DataFrame(pgm[i], columns=['PGM_i', ''], index=['i='+str(i),'']) for i in range(self.n_outcome)]
        pgm_pd = pd.concat(pgm_pd)
        tr_pgm_pd = [pd.DataFrame(['',np.real(np.trace(np.dot(rho[i], pgm[i])))], columns=['Tr[rho_i.PGM_i]'], index=['i='+str(i),'']) for i in range(self.n_outcome)]
        tr_pgm_pd = pd.concat(tr_pgm_pd)

        res_pd = pd.concat([rho_pd, E_pd, tr_E_pd, pgm_pd, tr_pgm_pd], axis=1)
        pd.options.display.float_format = '{:.4f}'.format #display decimals

        return res_pd





class SingleQubitPOVM():
    # Performing POVM on init_state
    def __init__(self, wires=None, dev=None):
        # wires
        self.wires = wires
        self.dev = dev


    def two_element_povm(self, params, wires):
        # First arbitrary unitary gate
        qml.Rot(params[0], params[1], params[2], wires=wires[0])

        # Controlled-RY gate controlled by first qubit in |0> state
        qml.PauliX(wires=wires[0])
        qml.CRY(params[3], wires=[wires[0],wires[1]])
        qml.PauliX(wires=wires[0])
        
        # Controlled-RY gate controlled by first qubit in |1> state
        qml.CRY(params[4], wires=[wires[0],wires[1]])


    def C2ePOVM(self, params, wires): 
        ctrl_2ePOVM = qml.ctrl(self.two_element_povm, control=wires[0])
        ctrl_2ePOVM(params, wires=[wires[1],wires[2]])


    def three_element_povm(self, params, wires): 
        # First arbitrary unitary gate
        qml.Rot(params[0], params[1], params[2], wires=wires[0])
        #qml.CRot(params[0], params[1], params[2], wires=[wires[1], wires[0]])

        # CC-RY gate controlled by |00> state 
        qml.PauliX(wires=wires[1])
        qml.PauliX(wires=wires[0])
        self.CCRY(params[3], wires=[wires[0], wires[1], wires[2]])
        qml.PauliX(wires=wires[0])

        # CC-RY gate controlled by |10> state
        self.CCRY(params[4], wires=[wires[0], wires[1], wires[2]])
        qml.PauliX(wires=wires[1])


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


    def KrausCirc(self, rank, params, wires):
        " two-element POVM module "
        self.two_element_povm(params=params[:5], wires=[wires[0], wires[-1]])
        
        if rank > 2:
            #self.three_element_povm(params[5:], wires=[self.wires[0], self.wires[1], self.wires[2]])
            qml.PauliX(wires=wires[-1])
            self.C2ePOVM(params[5:10], wires=[wires[-1], wires[0], wires[1]])
            qml.PauliX(wires=wires[-1])

        elif rank > 3:
            self.C2ePOVM(params[5:10], wires=[wires[-1], wires[0], wires[1]])
            #[10:15]

    # "Performing POVM on init_state(bloch_vec)"    
    def __call__(self, num_povms=2, init_circ=None, init_circ_wires=None):
        self.num = int(num_povms)
        self.init_circ = init_circ
        self.init_circ_wires = init_circ_wires
        
        
        return qml.QNode(self.povm_probs, self.dev)


    def povm_probs(self, params): 
        "State Preparation on circuit"
        self.init_circ(wires=self.init_circ_wires)
        
        self.KrausCirc(self.num, params, self.wires)

        # number of anciliary qubits for povm
        return qml.probs(wires=self.wires[1:])





class TwoQubitPOVM():
    # "Performing POVM on init_state(bloch_vec)"
    def __init__(self, wires=None, dev=None):
        # wires
        self.wires = wires
        self.dev = dev


    def SU4(self, params, wires):
        # two-qubit SU(4) gates designed by V. V. Shende et al., PRA 69 062321 (2004)
        qml.Rot(params[0], params[1], params[2], wires=wires[0])
        qml.Rot(params[3], params[4], params[5], wires=wires[1])

        qml.CNOT(wires=[wires[1], wires[0]])

        qml.RZ(params[6], wires=wires[0])
        qml.RY(params[7], wires=wires[1])

        qml.CNOT(wires=[wires[0], wires[1]])

        qml.RY(params[8], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])

        qml.Rot(params[9], params[10], params[11], wires=wires[0])
        qml.Rot(params[12], params[13], params[14], wires=wires[1])


    def CCRY(self, phi, wires):
        # 
        phi = phi/2
        qml.CRY(phi, wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CRY((-1)*phi, wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.CRY(phi, wires=[wires[0], wires[2]])


    def UCCRY(self, params, wires):
        qml.PauliX(wires=wires[0])
        qml.PauliX(wires=wires[1])
        self.CCRY(phi=params[0], wires=[wires[0], wires[1], wires[2]])
        qml.PauliX(wires=wires[1])
        qml.PauliX(wires=wires[0])
        
        qml.PauliX(wires=wires[0])
        self.CCRY(phi=params[1], wires=[wires[0], wires[1], wires[2]])
        qml.PauliX(wires=wires[0])

        qml.PauliX(wires=wires[1])
        self.CCRY(phi=params[2], wires=[wires[0], wires[1], wires[2]])
        qml.PauliX(wires=wires[1])

        self.CCRY(phi=params[3], wires=[wires[0], wires[1], wires[2]])


    def two_element_povm(self, params, wires):
        self.SU4(params[:15], wires=[wires[0],wires[1]])
        self.UCCRY(params[15:19], wires=[wires[0],wires[1],wires[2]])


    def C2ePOVM(self, params, wires):
        ctrl_2ePOVM = qml.ctrl(self.two_element_povm, control=wires[0])
        ctrl_2ePOVM(params, wires=[wires[1],wires[2],wires[3]])


    def four_element_povm(self, params, wires):
        self.two_element_povm(params[:19], wires=[wires[0],wires[1],wires[2]])
        qml.PauliX(wires=wires[2])
        self.C2ePOVM(params[19:38], wires=[wires[2],wires[0],wires[1],wires[3]])
        qml.PauliX(wires=wires[2])
        self.C2ePOVM(params[38:], wires=[wires[2],wires[0],wires[1],wires[3]])


    def KrausCirc(self, rank, params, wires):
        "Four-element POVMs"  
        self.two_element_povm(params[:19], wires=[wires[0],wires[1],wires[-1]])
        
        if rank > 2:
            qml.PauliX(wires=wires[-1])
            self.C2ePOVM(params[19:38], wires=[wires[-1],wires[0],wires[1],wires[2]])
            qml.PauliX(wires=wires[-1])

        if rank > 3:
            self.C2ePOVM(params[19:38], wires=[wires[-1],wires[0],wires[1],wires[2]])
            #[38:]
        """
        if self.num > 4:
            # Try!!
        """


    def __call__(self, num_povms=4, init_circ=None, init_circ_wires=None):
        self.num = int(num_povms)
        self.init_circ = init_circ
        self.init_circ_wires = init_circ_wires
        
        return qml.QNode(self.povm_probs, self.dev)


    def povm_probs(self, params):
        "State Preparation on circuit" 
        self.init_circ(wires=self.init_circ_wires)

        self.KrausCirc(self.num, params, self.wires)

        "num of ancillary qubits"
        self.num_ancilla_qubits = int(np.ceil(np.log(self.num) / np.log(2)))
        return qml.probs(wires=self.wires[2:])





class QuantumEmbedding():
    def __init__(self, wires, **kwargs):
        self.wires=wires

        self.embedding_type = kwargs['embed'] if 'embed' in kwargs.keys() else 'qfeat'
        if self.embedding_type == 'qfeat': 
            self.layer = kwargs['layer'] if 'layer' in kwargs.keys() else 2
            self.enc_type = kwargs['enc_type'] if 'enc_type' in kwargs.keys() else 'gaussian'


    def __call__(self, input_data):
        if self.embedding_type == 'qfeat':
            return self.QuantumFeatureMap(input_data)
        elif self.embedding_type == 'amplitude':
            return qml.AmplitudeEmbedding(features=input_data, wires=self.wires, normalize=True)

        

    def QuantumFeatureMap(self, input_data):
        wires = self.wires

        #parameter Feature Mapping
        features = self.encoding_fn(input_data, types=self.enc_type)

        for _ in range(self.layer):
            #XI
            qml.RX(features[0,0], wires=wires[0])

            #XX
            qml.Hadamard(wires=wires[0])
            qml.Hadamard(wires=wires[1])
            qml.CNOT(wires=[wires[0],wires[1]])
            qml.RZ(features[0,1], wires=wires[1])
            qml.CNOT(wires=[wires[0],wires[1]])
            qml.Hadamard(wires=wires[0])
            qml.Hadamard(wires=wires[1])

            #IX
            qml.RX(features[1,1], wires=wires[1])

            #ZX
            qml.Hadamard(wires=wires[1])
            qml.CNOT(wires=[wires[0],wires[1]])
            qml.RZ(features[1,2], wires=wires[1])
            qml.CNOT(wires=[wires[0],wires[1]])
            qml.Hadamard(wires=wires[1])

            #ZI
            qml.RZ(features[2,2], wires=wires[0])

            #ZZ
            qml.CNOT(wires=[wires[0],wires[1]])
            qml.RZ(features[2,3], wires=wires[1])
            qml.CNOT(wires=[wires[0],wires[1]])

            #IZ
            qml.RZ(features[3,3], wires=wires[1])

            #XZ
            qml.Hadamard(wires=wires[0])
            qml.CNOT(wires=[wires[0],wires[1]])
            qml.RZ(features[0,3], wires=wires[1])
            qml.CNOT(wires=[wires[0],wires[1]])
            qml.Hadamard(wires=wires[0])
        
        qml.RX(features[0,0], wires=wires[0])


    def encoding_fn(self, x, types='gaussian'):
        phi = np.diag(x)
        for i in range(len(x)):
            for j in range(1+i, len(x)):
                if types == 'pidiffmult':
                    phi[i,j] = (np.pi - x[i]) * (np.pi - x[j])
                elif types == 'mult':
                    phi[i,j] = np.pi * x[i] * x[j]
                elif types == 'binflipmult':
                    phi[i,j] = (np.pi/2) * (1-x[i]) * (1-x[j])
                elif types == 'gaussian':
                    phi[i,j] = np.exp((x[i]-x[j])**2/(8/np.log(np.pi)))
                elif types == 'invcoscos':
                    phi[i,j] = np.pi / (3 * np.cos(x[i]) * np.cos(x[j]))
                elif types == 'coscos':
                    phi[i,j] = np.pi * np.cos(x[i]) * np.cos(x[j])
                else:
                    raise ValueError("The available input 'types' is ['pidiffmult', 'mult', 'binflipmult', 'gaussian', 'invcoscos', 'coscos'].")
        return phi





class DataPOVMCirc:
    def __init__(self, num_measuring_qubits, num_povms):
        # number of qubits to measure
        self.num_measuring_qubits = int(num_measuring_qubits)

        # number of outcomes
        self.num_povms = num_povms
        self.num_ancilla_qubits = int(np.ceil(np.log(self.num_povms) / np.log(2)))

        
        
    
    def __call__(self, input_data, **kwargs):
        self._kwargs = kwargs
        self.input_data = input_data
        
        # number of data-embedded qubits
        self.num_data_qubits = int(np.ceil(np.log(len(input_data)) / np.log(2)))

        # merging circuit wires and povm wires
        self.wires = list(range(self.num_data_qubits+self.num_ancilla_qubits))
        self.povm_wires = self.wires[self.num_data_qubits-self.num_measuring_qubits:] # Wires for performing POVM

        dev = qml.device('default.qubit', wires=self.num_data_qubits+self.num_ancilla_qubits)        
        return qml.QNode(self.merging_circs, dev)


    def merging_circs(self, params):
        # State Preparation
        data_wires = list(range(self.num_data_qubits))
        quFeatmap = QuantumEmbedding(data_wires, enc_type=self._kwargs['enc_type'])
        quFeatmap(self.input_data)

        # POVM Preparation
        if self.num_measuring_qubits == 1:
            povm_circ = SingleQubitPOVM()

        elif self.num_measuring_qubits == 2:
            povm_circ = TwoQubitPOVM()
        
        povm_circ.KrausCirc(rank=self.num_povms, params=params, wires=self.povm_wires)

        # Probability distribution
        return qml.probs(wires=self.wires[self.num_data_qubits:])





class POVMbasedCLF():
    def __init__(self, num_measuring_qubits, num_povms):
        self.num_measuring_qubits = int(num_measuring_qubits) # number of qubits to measure
        self.num_povms = num_povms # number of outcomes
        
        # initial random parameters
        np.random.seed(20)
        # num_params: (2^n)^2-1 + 2^n : 
        num_params = (2**(2*num_measuring_qubits)-1+2**num_measuring_qubits) * (num_povms-1)
        self.povm_params = 2 * np.pi * np.random.random([num_params]) 

        
    def fit(self, X, y, **kwargs):
        self.enc_type = kwargs['enc_type'] if 'enc_type' in kwargs.keys() else 'gaussian'
        
        # Prepare QNodeCollection
        self.qnodes = qml.QNodeCollection()

        for x in X:
            circ = DataPOVMCirc(self.num_measuring_qubits, self.num_povms)
            qnode = circ(x, enc_type=self.enc_type)
            self.qnodes.append(qnode)

        # Find a priori probabilities by conuting y_train
        self.a_priori_probs = [1 / len(y)] * len(y)
        self._y = y

        # Training POVM circuit to become a classifier.
        # print(self.qnodes(self.povm_params))
        self.run_opt()


    def cost_fn(self, params, types='err'):
        #print(types)
        probs_povm = self.qnodes(params)
        q = self.a_priori_probs

        res = 0 if types == 'suc' else 1
        
        for i in range(len(q)):

            if types == 'err':
                res -= q[i] * probs_povm[i][self._y[i]]
                #res -= q[i] * probs_povm[i][y_rev[i]]
            elif types == 'inc':
                res -= q[i] * sum(probs_povm[i][:(self.num_povms-1)])
            elif types == 'suc':
                res += q[i] * (sum(probs_povm[i]) - probs_povm[i][self._y[i]])

        return res


    def run_opt(self, steps=3000):
        # initialize the optimizer
        #opt = qml.GradientDescentOptimizer(stepsize=0.4)
        opt = qml.AdamOptimizer(stepsize=0.02)
        #opt = qml.AdagradOptimizer(stepsize=0.2)

        # cost fn for initial random params
        cost_list = [self.cost_fn(self.povm_params)]
        print("Cost(init_params)    : {: .7f}".format(cost_list[0]))

        # update the circuit parameters
        for i in range(steps):     # set the number of steps
            self.povm_params = opt.step(self.cost_fn, self.povm_params)
            cost_list.append(self.cost_fn(self.povm_params))
            if (i+1) % 20 == 0:
                print("Cost after step {:4d} : {: .7f}".format(i + 1, cost_list[i + 1]))

            if i>1 and cost_list[i-1] - cost_list[i] < 1e-7:
                break
        #print("Optimized rotation angles: {}".format(self.povm_params))
        #return cost_list
        self.cost_convg = cost_list

    
    def predict(self, X_test):
        # Prepare QNodeCollection
        qnodes_predict = qml.QNodeCollection()

        for x in X_test:
            circ = DataPOVMCirc(self.num_measuring_qubits, self.num_povms)
            qnode = circ(x, enc_type=self.enc_type)
            qnodes_predict.append(qnode)

        self.probs_predict = qnodes_predict(self.povm_params)

        return np.array(np.argmax(self.probs_predict, axis=1).tolist())



#위의 twoqubitPOVM class에서는 input state qubit이 무조건 2개로 고정!
#아래의 classificatino에서는 data qubits이랑 povm 측정할 qubit의 갯수가 불일치해도 괜찮게 작성하기!
class POVM_iris():
    def __init__(self, num_measuring_qubits, num_povms):
        # number of data-encoded qubits
        self.num_measuring_qubits = int(num_measuring_qubits)

        # number of outcomes
        self.num_povms = num_povms
        self.num_ancilla_qubits = int(np.ceil(np.log(self.num_povms) / np.log(2)))

        # initial random parameters 
        np.random.seed(20)
        # num_params: (2^n)^2-1 + 2^n : 
        num_params = (2**(2*self.num_measuring_qubits)-1+2**self.num_measuring_qubits) * (self.num_povms-1)
        self.povm_params = 2 * np.pi * np.random.random([num_params]) 

        # Prepare QNodeCollection
        self.qnodes = qml.QNodeCollection()

    
    def fit(self, X, y, **kwargs):

        # the number of data-encoded qubits
        self.num_data_qubits = int(np.ceil(np.log(len(X[0])) / np.log(2)))

        # Declare total_wires and povm_wires
        self.wires = list(range(self.num_data_qubits+self.num_ancilla_qubits))
        self.povm_wires = self.wires[self.num_data_qubits-self.num_measuring_qubits:]

        for i in range(len(X)):
            cdev = qml.device('default.qubit', wires=self.num_data_qubits+self.num_ancilla_qubits)

            # qnode
            qnode = self.quantemb_povm_qnode(X[i], cdev)
            print(i)
            # print(qnode(self.povm_params))
            self.qnodes.append(qnode)

        # Find a priori probabilities by conuting y_train
        self.a_priori_probs = [1 / len(y)] * len(y)
        self._y = y

        # Training POVM circuit to become a classifier.
        #self.run_opt()
        print(self.qnodes(self.povm_params))


    def quantemb_povm_qnode(self, input_data, dev): 
        self.datapoint = input_data
        print(self.datapoint)
        return qml.QNode(self.quantemb_povm_circ, dev)

    
    def quantemb_povm_circ(self, params):

        # Quantum Embedding
        if self.enc_type == 'amplitude':
            qml.AmplitudeEmbedding(features=self.datapoint, wires=self.wires[:self.num_data_qubits], normalize=True)
        
        elif self.enc_type == 'qfeat':
            self.quantum_feature(wires=self.wires, input_data=self.datapoint)

        # POVM circuit
        if self.num_measuring_qubits == 1:
            povm_circ = SingleQubitPOVM()

        elif self.num_measuring_qubits == 2:
            povm_circ = TwoQubitPOVM()
            
        povm_circ.KrausCirc(rank=self.num_povms, params=params, wires=self.povm_wires)

        # Probability distribution
        return qml.probs(wires=self.wires[self.num_data_qubits:])


    def cost_fn(self, params, types='err'):
        #print(types)
        probs_povm = self.qnodes(params)
        q = self.a_priori_probs

        res = 0 if types == 'suc' else 1
        
        #bin_len = int(np.ceil(np.log(self.num)/ np.log(2)))
        y_rev = self._dec_2_bin()
        for i in range(len(q)):
            """
            b2d = format(self._y[i], '0'+str(bin_len)+'b')
            inverse = b2d[::-1]
            d2b = int(inverse, 2)
            """
            if types == 'err':
                res -= q[i] * probs_povm[i][y_rev[i]]
            elif types == 'inc':
                res -= q[i] * sum(probs_povm[i][:(self.num_povms-1)])
            elif types == 'suc':
                res += q[i] * (sum(probs_povm[i]) - probs_povm[i][y_rev[i]])
        """
        if types == 'mcm':
            i=2
            tr_E_i_rho = q * probs_povm[:, y_rev[i]]
            res -= (tr_E_i_rho[i] / sum(tr_E_i_rho))
        """

        return res
        
    
    def _dec_2_bin(self, x=None):
        if x == None: 
            x = self._y

        bin_len = int(np.ceil(np.log(self.num_povms)/ np.log(2)))

        d2b_list = []
        for i in range(len(x)):
            
            b2d = format(x[i], '0'+str(bin_len)+'b')
            inverse = b2d[::-1]
            d2b = int(inverse, 2)
            d2b_list.append(d2b)

        return d2b_list


    def run_opt(self, steps=3000):
        # initialize the optimizer
        #opt = qml.GradientDescentOptimizer(stepsize=0.4)
        opt = qml.AdamOptimizer(stepsize=0.02)
        #opt = qml.AdagradOptimizer(stepsize=0.2)

        # cost fn for initial random params
        cost_list = [self.cost_fn(self.povm_params)]
        print("Cost(init_params)    : {: .7f}".format(cost_list[0]))

        # update the circuit parameters
        for i in range(steps):     # set the number of steps
            self.povm_params = opt.step(self.cost_fn, self.povm_params)
            cost_list.append(self.cost_fn(self.povm_params))
            if (i+1) % 20 == 0:
                print("Cost after step {:4d} : {: .7f}".format(i + 1, cost_list[i + 1]))

            if i>1 and cost_list[i-1] - cost_list[i] < 1e-7:
                break
        #print("Optimized rotation angles: {}".format(self.povm_params))
        #return cost_list
        self.cost_convg = cost_list


    def cost_fn_mcm(self, params, k):
        probs_povm = self.qnodes(params, parallel=True)
        q = self.a_priori_probs

        res = 0 
        y_rev = self._dec_2_bin()
                    
        #i=0
        tr_E_k_rho = q * probs_povm[:, y_rev[k]]
        res = (-1)*(tr_E_k_rho[k] / sum(tr_E_k_rho))

        return res


    def run_opt_mcm(self, steps=3000):
        # initialize the optimizer
        #opt = qml.GradientDescentOptimizer(stepsize=0.01)
        opt = [qml.AdamOptimizer()]*3

        # cost fn for initial random params
        cost0_list = [self.cost_fn_mcm(self.povm_params, 0)]
        cost1_list = [self.cost_fn_mcm(self.povm_params, 1)]
        cost2_list = [self.cost_fn_mcm(self.povm_params, 2)]

        print("Cost(init_params)    : {: .7f} {: .7f} {: .7f}".format(cost0_list[0], cost1_list[0], cost2_list[0]))

        # update the circuit parameters
        for i in range(steps):    
            self.povm_params = opt[0].step(self.cost_fn_mcm, self.povm_params, k=0)
            self.povm_params = opt[1].step(self.cost_fn_mcm, self.povm_params, k=1)
            self.povm_params = opt[2].step(self.cost_fn_mcm, self.povm_params, k=2)

            cost0_list.append(self.cost_fn_mcm(self.povm_params, 0))
            cost1_list.append(self.cost_fn_mcm(self.povm_params, 1))
            cost2_list.append(self.cost_fn_mcm(self.povm_params, 2))

            if (i+1) % 20 == 0:
                print("Cost after step {:5d}: {: .7f} {: .7f} {: .7f}".format(i+1, cost0_list[i+1], cost1_list[i+1], cost2_list[i+1]))

            if i>1 and cost0_list[i-1] - cost0_list[i] < 1e-7 \
                    and cost1_list[i-1] - cost1_list[i] < 1e-7 \
                    and cost2_list[i-1] - cost2_list[i] < 1e-7:
                break

        self.cost_convg = [cost0_list, cost1_list, cost2_list]


    def predict(self, X_test):
        qnodes_predict = qml.QNodeCollection()

        for x in X_test:
            cdev = qml.device('default.qubit', wires=self.num_data_qubits+self.num_ancilla_qubits) #, shots=1)

            povm_circ = TwoQubitPOVM(wires=self.povm_wires, dev=cdev) 
        
            # Construct povm circuit whose output is the probability of a measurement outcome on ancilla qubits
            if self.enc_type == 'amplitude':
                qnode = povm_circ(self.num_povms, self.amplitude_encoding, x) # init_circ_params = X[i]
            
            elif self.enc_type == 'qfeat':
                qnode = povm_circ(self.num_povms, self.quantum_feature, x)


            qnodes_predict.append(qnode)
        
        self.probs_predict = qnodes_predict(self.povm_params) # parallel=True It becomes slow when it's "True"

        return np.array(self._dec_2_bin(np.argmax(self.probs_predict, axis=1).tolist()))





def affine_preprocessing(data, minmax='m1-to-p1'):
    # input: list of data
    maxvals, minvals = np.max(data, axis=0), np.min(data, axis=0)
    if minmax=='m1-to-p1':
        #slope
        slopes = 2 / (maxvals-minvals) 
        #intercepts
        maxvals, minvals = maxvals * slopes, minvals * slopes
        intercepts = np.median([maxvals, minvals], axis=0)

    elif minmax=='0-to-2pi':
        slopes = 2*np.pi / (maxvals-minvals) 
        #intercepts
        intercepts = minvals * slopes
        
    return data * slopes - intercepts


def pos_affine_preprocessing(data):
    # input: list of data
    maxvals, minvals = np.max(data, axis=0), np.min(data, axis=0)
    slopes = 1 / (maxvals-minvals)
    maxvals, minvals = maxvals * slopes, minvals * slopes
    #intercepts = np.median([maxvals, minvals], axis=0)
    return data * slopes - minvals#- intercepts


def normalize_preprocessing(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)





class TQ_POVM_Clf():
    def __init__(self, n, wires, a_priori_probs, input_circ, input_circ_params=None):
        # number of outcomes
        self.n_outcome = n

        # wires
        self.wires = wires
        devs = [qml.device('default.qubit', wires=len(wires))] * n

        # Prior Probabilities for each state
        self.a_priori_probs = a_priori_probs

        # initial random parameters
        np.random.seed(20) #seed(1)
        self.povm_params = 2 * np.pi * np.random.random([19*(n-1)])

        self.qnodes = qml.QNodeCollection()
        for i in range(self.n_outcome):
            povm_circ = TwoQubitPOVM(wires=wires, dev=devs[i])       

            # Construct povm circuit whose output is the probability of a measurement outcome on ancilla qubits
            qnode = povm_circ(n, input_circ[i], input_circ_params=input_circ_params) 
            # print(qnode(self.povm_params))
            self.qnodes.append(qnode)


    def cost_fn(self, x):
        probs_povm = self.qnodes(x, parallel=True)
        q = self.a_priori_probs

        res = 1
        bin_len = int(len(self.wires[2:]))
        for i in range(self.n_outcome):
            b2d = format(i, '0'+str(bin_len)+'b')
            inverse = b2d[::-1]
            d2b = int(inverse, 2)
            
            res -= q[i] * probs_povm[i][d2b]
            #print(i, d2b, probs_povm[i][d2b])
        return res


    def run_opt(self, steps=3000):
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

            if i>1 and cost_list[i-1] - cost_list[i] < 1e-7:
                break
        #print("Optimized rotation angles: {}".format(self.povm_params))
        return cost_list


    def SU4_matrices(self, params): # 15 params
        get_matrix = qml.transforms.get_unitary_matrix(TwoQubitPOVM(n=4).SU4, wire_order=[0,1])
        return get_matrix(params, wires=[0,1])


    def diag_matrices_from_RY(self, params): # 4 params
        D_cos = np.diag(np.cos([params[i] for i in range(4)]))
        D_sin = np.diag(np.sin([params[i] for i in range(4)]))
        return D_cos, D_sin


    def povm_matrices(self):
        U_su4 = []
        D_cos = []
        D_sin = []
        for i in range(3):
            U_su4.append(self.SU4_matrices(self.povm_params[19*i:19*i+15]))
            res_D_cos, res_D_sin = self.diag_matrices_from_RY(self.povm_params[19*i+15:19*(i+1)])
            D_cos.append(res_D_cos)
            D_sin.append(res_D_sin)

        modules_cos = [np.dot(D_cos[i], U_su4[i]) for i in range(3)]
        modules_sin = [np.dot(D_sin[i], U_su4[i]) for i in range(3)]
        
        K0 = np.dot(modules_cos[1], modules_cos[0])
        K1 = np.dot(modules_sin[1], modules_cos[0])
        K2 = np.dot(modules_cos[2], modules_sin[0])
        K3 = np.dot(modules_sin[2], modules_sin[0])

        E0 = np.real(np.dot(K0.conjugate().transpose(), K0))
        E1 = np.real(np.dot(K1.conjugate().transpose(), K1))
        E2 = np.real(np.dot(K2.conjugate().transpose(), K2))
        E3 = np.real(np.dot(K3.conjugate().transpose(), K3))

        return E0, E1, E2, E3





def Helstrom(q, rho):
    try:
        len(rho) == len(rho[0]) == 2
    except:
        ValueError('Only two qubits can be discriminated!')
    
    # Spectral Decomposition for q_0*rho_0 - q_1*rho_1
    lambdas, povm_bases = np.linalg.eig(q[0]*rho[0] - q[1]*rho[1])

    Helstrom_bound = (1 - np.sum(np.abs(lambdas))) / 2
    optimal_povms = [np.outer(povm_bases[:,i], np.conj(povm_bases[:,i])) for i in range(2)]

    return Helstrom_bound, optimal_povms


# Pretty Good Measurement
def pgm(q, rho): 
    num = len(rho)
    rho_tot = np.sum([q[i] * rho[i] for i in range(num)], axis=0)
    rho_inv_sqrt = np.linalg.inv(sqrtm(rho_tot))
    res = [np.dot(np.dot(rho_inv_sqrt, q[i] * rho[i]),rho_inv_sqrt) for i in range(num)]

    med = 1 - np.sum([q[i] * np.trace(np.dot(rho[i], res[i])) for i in range(num)])
    return np.real(med), res


def fidelity_matrix(state):
    # data : list of vecs : # of vecs X dim of vec
    num_state = len(state)
    dim = len(state[0])
    
    
    res = np.empty((num_state, num_state))
    for i in range(num_state):
        for j in range(num_state):
            if len(state[0].shape) == 1:
                res[i][j] = np.abs(np.dot(state[i].conj(), state[j]))**2
            else: 
                res[i][j] = np.trace(sqrtm(np.dot(state[i], state[j])))**2
    
    return res


def fidelity_ratio(data):
    return 0





class POVM_Clf_SDP():
    def __init__(self, dim_povm=2, num_povms=2, problem_type='Primal'):
        self.problem = problem_type
        self.dim = dim_povm
        self.num = num_povms


    def Primal(self):
        # Create num dim x dim matrix variables
        E = [cp.Variable((self.dim, self.dim), hermitian=True) for x in range(self.num)]

        # Create constraints
        ## Equality Constraints
        sum_all_E = 0
        for i in range(self.num):
            sum_all_E += E[i]
        constraints = [sum_all_E == np.eye(self.dim)]
        ## Inequality Constraints
        constraints += [
            E[i] >> 0 for i in range(self.num)
        ]

        # Form an objective function.
        obj = 0
        for i in range(self.num):
            obj += cp.real(cp.trace(E[i] @ self.q_rho[i]))

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve()

        print("Opt is Done. \nStatus:", prob.status)
        E_opt = [E[i].value for i in range(self.num)]
        med_value = 1 - prob.value

        return med_value, E_opt


    def Dual(self):

        K = cp.Variable((self.dim,self.dim), hermitian=True)
        constraints = [K - self.q_rho[i] >> 0 for i in range(self.num)]
        prob = cp.Problem(cp.Minimize(cp.real(cp.trace(K))), constraints)
        prob.solve()

        print("Opt is Done. \nStatus:", prob.status)
        K_opt = K.value
        med_value = 1 - prob.value

        return med_value, K_opt


    def __call__(self, init_states, a_priori_probs):
        
        if self.dim != len(init_states[0]):
            raise ValueError("The dimension of POVM != The dimension of quantum state")
        """
        elif self.num != len(init_states):
            raise ValueError("The number of POVMS != The number of quantum states")
        """
        
        
        if len(init_states[0]).shape == 1:
            self.init_rhos = [np.outer(init_states[i], np.conj(init_states[i])) for i in range(self.num)]
        #else: density state
        self.q_rho = [a_priori_probs[i] * self.init_rhos[i] for i in range(self.num)]

        return self.Primal() if self.problem == 'Primal' else self.Dual()





def dp_res(rho, **kwargs):
    num = len(rho)

    rho_pd = [pd.DataFrame(rho[i], columns=['rho_i', ''], index=['i='+str(i),'']) for i in range(num)]
    rho_pd = pd.concat(rho_pd)

    pd_list = [rho_pd]
    for key, val in kwargs.items():
        M_pd = [pd.DataFrame(val[i], columns=[key+'_i', ''], index=['i='+str(i),'']) for i in range(num)]
        M_pd = pd.concat(M_pd)
        pd_list.append(M_pd)

        tr_M_pd = [pd.DataFrame(['',np.real(np.trace(np.dot(rho[i], val[i])))], columns=['Tr[rho_i.'+key+'_i]'], index=['i='+str(i),'']) for i in range(num)]
        tr_M_pd = pd.concat(tr_M_pd)
        pd_list.append(tr_M_pd)
    
    res_pd = pd.concat(pd_list, axis=1)
    pd.options.display.float_format = '{:.4f}'.format #display decimals

    return res_pd




def povm_validity(E_opt, atol=1e-6):
    num_povms = len(E_opt)
    dim_povm = len(E_opt[0])

    column_name = []
    for i in range(num_povms):
        column_name += ['']*(dim_povm-1) + ['E_'+str(i)]

        pd.options.display.float_format = "{:,.5f}".format
    df1 = pd.DataFrame(np.array(E_opt).reshape(num_povms*dim_povm,dim_povm).T, columns=column_name, index=['POVM','matrices']+['']*(dim_povm-2))


    Completeness = [str(np.all(np.isclose(np.sum(E_opt, axis=0), np.eye(dim_povm), atol=atol)))] + ['']*(num_povms-1)
    Hermicity = []
    Positive_semidefinite = []
    for i in range(num_povms):
        Hermicity.append(str(np.all(np.isclose(E_opt[i].conjugate().transpose(), E_opt[i], atol=atol))))
        
        val, _ = np.linalg.eigh(E_opt[i])
        Positive_semidefinite.append(str(np.all(val >= (-1)*atol)))

    column_name = ['E_'+str(i) for i in range(num_povms)]
    df2 = pd.DataFrame(np.array([Hermicity, Positive_semidefinite, Completeness]), columns=column_name, index=['Hermicity', 'Positive-semidefinite', 'Completeness'])
    
    display(df1, df2)



def dp_probs(state, measure, dp_table=True):
    num_state = len(state)
    num_povm = len(measure)
    q = [1/num_state] * num_state
    probs = np.empty((num_state,num_povm))
    for s in range(num_state):
        for m in range(num_povm):
            if len(state[s].shape) == 1: # Pure state
                probs[s][m] = np.real(np.dot(np.dot(state[s].conj(), measure[m]), state[s]))
            else: 
                probs[s][m] = np.real(np.trace(np.dot(state[s], measure[m])))
            
    probs = np.hstack((probs, np.sum(probs, axis=1)[:,np.newaxis]))

    if dp_table:
        df = pd.DataFrame(data=probs, 
                    columns=[['Measure']*(num_povm+1), list(range(num_povm))+ ['sum_probs']],
                    index=[['State']*num_state, list(range(num_state))])
        df.style.set_caption('Probabilities of the outcome j for given state i')
        return df
    else:
        return probs   

"""
dev_qiskit = qml.device("qiskit.aer", wires=2)
@qml.qnode(dev_qiskit)
def circ_test(name):
    
    #exec(name, globals())
    name(wires=[0,1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


dev_qiskit1 = qml.device("qiskit.aer", wires=2)
@qml.qnode(dev_qiskit1)
def circ_test1(name):
    
    exec(name, globals())
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))



def test_fn(name, num):
    name(num)

def test1_fn(name_list):
    
    for name in name_list:
        name(1)
"""

def sdp_med(states, num_povm=3):
    num_state = len(states)
    a_priori_probs = [1/num_state] * num_state
    init_rhos = []
    for i in range(num_state):
        rho = np.outer(states[i], np.conj(states[i])) if len(states[i].shape)==1 else states[i]
        init_rhos.append(rho) 
    q_rho = [a_priori_probs[i] * init_rhos[i] for i in range(num_state)]
    
    dim = len(states[0])

    E = [cp.Variable((dim, dim), hermitian=True) for _ in range(num_povm)]

    # Create constraints
    ## Equality Constraints
    sum_all_E = 0
    for i in range(num_povm):
        sum_all_E += E[i]
    constraints = [sum_all_E == np.eye(dim)]
    ## Inequality Constraints
    constraints += [
        E[i] >> 0 for i in range(num_povm)
    ]

    # Form an objective function.
    obj = 1
    for i in range(num_state):
        obj -= cp.real(cp.trace(E[i] @ q_rho[i]))


    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()

    print("Opt is Done. \nStatus:", prob.status)
    E_opt = [E[i].value for i in range(num_povm)]
    med_value = prob.value
    print("Optimal Value:", med_value)

    return med_value, E_opt


def sdp_mcm(states, num_povm=3):
    num_state = len(states)
    a_priori_probs = [1/num_state] * num_state
    init_rhos = [np.outer(states[i], np.conj(states[i])) for i in range(num_state)]
    normalized_rhos, _ = pgm(a_priori_probs, init_rhos)

    dim = len(states[0])

    Q = [cp.Variable((dim, dim), hermitian=True) for _ in range(num_povm)]

    med_value = []
    Q_opt = []
    status = []
    for i in range(num_povm):
        # Create constraints
        ## Equality Constraints
        constraints = [cp.real(cp.trace(Q[i])) == 1] #np.eye(dim)]
        ## Inequality Constraints
        constraints += [Q[i] >> 0]

        # Form an objective function.

        obj = cp.real(cp.trace(normalized_rhos[i] @ Q[i]))

        prob = cp.Problem(cp.Maximize(obj), constraints)
        prob.solve()

        #print("Opt is Done. \nStatus:", prob.status)
        status.append(prob.status)
        Q_opt.append(Q[i].value)
        med_value.append(prob.value)


    return med_value, Q_opt 


def plot_ROC_curve(y_tests, y_scores):
    n_classes = len(y_scores[0]) 
    n_classes_ = n_classes if len(y_scores[0]) > 2 else 3
    y_tests = label_binarize(y_tests, classes=range(n_classes_))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_tests[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,8))
    lw = 2
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            #color="darkorange",
            lw=lw,
            label="Class %i (AUC = %0.2f)" % (i, roc_auc[i]),
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=15)
    plt.ylim([0.0, 1.05])
    plt.yticks(fontsize=15)
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title("ROC curves by using probability distribution", fontsize=20)
    plt.legend(loc="lower right", fontsize=15)
    plt.show()


def plot_mean_ROC_curve(y_tests, y_scores, n_classes=3, n_splits=5):
    #n_classes = len(y_scores[0][0])
    #n_splits = len(y_scores)

    mean_fpr = np.linspace(0, 1, 100)
    tprs = [[] for _ in range(n_classes)] # tprs[class][k-fold]
    aucs = [[] for _ in range(n_classes)] # aucs[class][k-fold]
    
    for k in range(n_splits): 
        fpr, tpr, roc_auc = roc_curve_mc(y_tests[k], y_scores[k][:,:3])

        for l in range(n_classes):
            interp_tpr = np.interp(mean_fpr, fpr[l], tpr[l]) 
            interp_tpr[0] = 0.0
            tprs[l].append(interp_tpr)    
            aucs[l].append(roc_auc[l])

    mean_tprs = dict()
    for l in range(n_classes):
        mean_tprs[l] = np.mean(np.array(tprs[l]), axis=0)

    mean_aucs = np.mean(np.array(aucs), axis=1)

    #Plotting
    plt.figure(figsize=(8,8))
    lw = [2,3,4]
    colors = ['blue','red', 'green']
    linestyle = ['solid', 'dashed', 'dotted']
    for i in range(n_classes):
        plt.plot(
            mean_fpr,
            mean_tprs[i],
            color=colors[i],
            lw=lw[i],
            linestyle=linestyle[i],
            label="Class %i (Mean AUC = %0.3f)" % (i, mean_aucs[i]),
        )
    plt.plot([0, 1], [0, 1], color="lightgrey", lw=2, linestyle=(0, (3, 5, 1, 5)))
    plt.xlim([-0.03, 1.03])
    plt.xticks(fontsize=20)
    plt.ylim([-0.03, 1.03])
    plt.yticks(fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=25)
    plt.ylabel("True Positive Rate", fontsize=25)
    plt.title("Mean ROC Curves", fontsize=28)
    plt.legend(loc="lower right", fontsize=18)
    plt.savefig('./data/meanROCcurves.png', bbox_inches='tight', transparent=True)
    plt.show()


def roc_curve_mc(y_tests, y_scores):
    n_classes = len(y_scores[0]) 
    n_classes_ = n_classes if len(y_scores[0]) > 2 else 3
    y_tests = label_binarize(y_tests, classes=range(n_classes_))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_tests[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc