import numpy as np
import numba as nb

import cirq

import openfermion as of
from openfermion.testing.testing_utils import random_quadratic_hamiltonian
from openfermion.ops import QuadraticHamiltonian
import math
from multiprocessing import Pool


def c_k(k):
    # if k > 2 * len(qubits):
    #     raise ValueError("k cannot be more than 2n")

    qubits = cirq.LineQubit.range(k)

    Zs = [cirq.Z(q) for q in qubits]
    Xs = [cirq.X(q) for q in qubits]
    Ys = [cirq.Y(q) for q in qubits]

    j = math.ceil(k/2)

    j_gate = Xs[j-1] if k % 2 == 1 else Ys[j-1]

    return cirq.PauliString([Zs[i] for i in range(j-1)] + [j_gate])

def schatten_norm_pow(A, p):
    '''
    Computes the modified Schatten norm to the p power

    @param: A: matrix to take the norm of
    @param: p: the order for the Schatten norm
    '''
    # Compute the singular values
    singular_values = np.linalg.svd(A, compute_uv=False)
    
    # Raise the singular values to the power of p, sum them, and take the p-th root
    if p == np.inf:
        # The Schatten infinity norm is just the maximum singular value
        return np.max(singular_values)
    else:
        # Compute the sum of the singular values raised to the power p
        return np.sum(singular_values**p)

def covariance_jk_matrix(density_matrix, j, k):
    if j == k:
        return 0
    
    qubits = cirq.LineQubit.range(int(np.log2(len(density_matrix)))) 

    cj = c_k(j)
    ck = c_k(k)

    circuit = cirq.Circuit()
    circuit.append([cirq.I(q) for q in qubits])
    circuit.append([cj*ck])

    expectation = np.trace((density_matrix @ circuit.unitary()))

    return 1j * expectation

def covariance_matrix(denisty_matrix):
    n = int(np.log2(len(denisty_matrix)))

    covariance_mat = np.array([[covariance_jk_matrix(denisty_matrix, j=j, k=k) for j in range(1,2*n + 1)] for k in range(1,2*n + 1)])
    return covariance_mat

def nongaussianity_matrix(density_matrix, alpha):
    '''
    @param: density_matrix: desity matrix of the state to examine
    @param: alpha: the order for the schatten norm
    '''
    cm = covariance_matrix(density_matrix)
    norm_power = schatten_norm_pow(cm, alpha)

    n = int(np.log2(density_matrix.shape[0]))
    return n - norm_power/2

def covariance_jk(state_vector, j, k):
    if j == k:
        return 0
    
    qubits = cirq.LineQubit.range(int(np.log2(len(state_vector)))) 

    cj = c_k(j)
    ck = c_k(k)

    circuit = cirq.Circuit()
    circuit.append([cirq.I(q) for q in qubits])
    circuit.append([cj*ck])

    expectation = state_vector.conj() @ (circuit.unitary() @ state_vector)

    return 1j * expectation

def covariance_jk_circ(state_vector, j, k):
    if j == k:
        return 0
    
    qubits = cirq.LineQubit.range(int(np.log2(len(state_vector)))) 

    cj = c_k(j)
    ck = c_k(k)

    circuit = cirq.Circuit()
    circuit.append([cirq.I(q) for q in qubits])
    circuit.append([cj*ck])
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)

    expectation = state_vector.conj() @ (result.final_state_vector)

    return 1j * expectation

def covariance(state_vec):
    n = int(np.log2(len(state_vec)))

    covariance_mat = np.array([[covariance_jk(state_vec, j=j, k=k) for j in range(1,2*n + 1)] for k in range(1,2*n + 1)])
    return covariance_mat


def nongaussianity(state_vec, alpha):
    '''
    @param: density_matrix: desity matrix of the state to examine
    @param: alpha: the order for the schatten norm
    '''
    cm = covariance(state_vec)
    norm_power = schatten_norm_pow(cm, alpha)

    n = int(np.log2(state_vec.shape[0]))
    return n - norm_power/2



# sum of squares from stackexchange https://stackoverflow.com/questions/56047264/generate-two-random-numbers-whose-square-sum-1
from functools import reduce
from math import sqrt
from random import gauss, uniform
rng = np.random.default_rng()

def sum_of_squares_is_one(n = 2):
    if ((n < 2) or (int(n) != n)) is True:
        raise Exception("Invalid argument for n")
    l = [gauss(0.0, 1.0) for _ in range(n)]
    norm = sqrt(reduce(lambda sum,x: sum + x*x, l, 0.0)) # / uniform(0.95, 1.05)
    return [x / norm for x in l]


def superposition(states):
    '''
        Generates a superposition of the given states
        @param states: a list of quantum states in state vector form
    '''

    n = len(states)
    coeffs = cirq.testing.random_superposition(n)
    states = np.asarray(states)

    modified_state_vector = np.sum([states[i]*coeffs[i] for i in range(n)], axis=0)

    # print(n, coeffs, states)
    # print(sum(np.array(coeffs)**2))
    return modified_state_vector/np.linalg.norm(modified_state_vector)

def generate_random_gaussian_state_vector(norbs):
    qh = random_quadratic_hamiltonian(norbs, conserves_particle_number=True, real=True, expand_spin=False)
    gspc = of.circuits.prepare_gaussian_state(
        qubits=cirq.LineQubit.range(0,norbs), quadratic_hamiltonian=qh, occupied_orbitals=None
    )
    circuit = cirq.Circuit()
    circuit.append(gspc)
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    return result.final_state_vector

# this one is slower
# def generate_random_gaussian_state_vector(norbs):
#     qh = random_quadratic_hamiltonian(norbs, conserves_particle_number=True, real=True, expand_spin=False)
#     _g_energy, g_state = of.circuits.jw_get_gaussian_state(qh, occupied_orbitals=None)
#     return g_state

def generate_n_samples(n, norbs):
    # return np.fromfunction(lambda x: random_quadratic_hamiltonian(norbs, conserves_particle_number=True, real=True, expand_spin=False), shape=(n,))
    states = [generate_random_gaussian_state_vector(norbs) for _ in range(n)]
    N = 2**norbs
    s = [np.pad(state, (0, N - len(state))) for state in states]

    return s

def sample_superposition(n, norbs):
    return superposition(generate_n_samples(n, norbs))

# def generate_n_samples(n, norbs):
#     # return np.fromfunction(lambda x: random_quadratic_hamiltonian(norbs, conserves_particle_number=True, real=True, expand_spin=False), shape=(n,))
#     pool = Pool()
#     states = pool.map(generate_random_gaussian_state_vector, np.full(n, norbs))
#     return states

# def generate_n_samples(n, norbs):
#     # return np.fromfunction(lambda x: random_quadratic_hamiltonian(norbs, conserves_particle_number=True, real=True, expand_spin=False), shape=(n,))
#     states = [generate_random_gaussian_state_vector(norbs) for _ in range(n)]
#     return states

# def single_nongaussianity(n, norbs):
#     def ng(alpha):
#         return nongaussianity(superposition(generate_n_samples(n, norbs)), alpha)
#     return ng

def single_nongaussianity(n, norbs, alpha):
    return nongaussianity(superposition(generate_n_samples(n, norbs)), alpha)

def nongaussianity2(state_vec):
    return nongaussianity(state_vec, 2)
