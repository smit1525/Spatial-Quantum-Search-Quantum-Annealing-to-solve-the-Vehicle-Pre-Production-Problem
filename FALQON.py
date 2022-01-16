
! pip install networkx

import networkx as nx
from networkx.utils import arbitrary_element

from networkx import DiGraph

G = DiGraph()

# Layer 1
G.add_node(0, level="and")

# Layer 2
G.add_node(1, label="and")
G.add_node(2, label="and")
G.add_node(3, label="and")
G.add_node(4, label="and")
G.add_node(5, label="and")
G.add_node(6, label="and")
G.add_node(7, label="and")
G.add_node(8, label="and")
G.add_node(9, label="and")
G.add_node(10, label="and")


G.add_node(11, label="or")
G.add_node(12, label="or")
G.add_node(13, label="or")
G.add_node(14, label="or")
G.add_node(15, label="or")

G.add_edge(0,1)
G.add_edge(0,2)
G.add_edge(0,3)
G.add_edge(0,4)
G.add_edge(0,6)
G.add_edge(0,8)

# Layer 3

G.add_node(16, level="b1")
G.add_node(17, level="b1")
G.add_node(18, level="b1")
G.add_node(19, level="b1")
G.add_node(20, level="b1")
G.add_node(21, level="b1")
G.add_node(22, level="b1")
G.add_node(23, level="b1")
G.add_node(24, level="b1")
G.add_node(25, level="not")
G.add_node(26, level="bi1")
G.add_node(27, level="bi2")
G.add_node(28, level="bi4")
G.add_node(29, level="bi5")
G.add_node(30, level="ci1")
G.add_node(31, level="ci2")


G.add_edge(1,11)
G.add_edge(1,12)
G.add_edge(2,12)
G.add_edge(2,13)
G.add_edge(3,14)
G.add_edge(3,15)
G.add_edge(4,15)
G.add_edge(4,5)
G.add_edge(6,5)
G.add_edge(6,7)
G.add_edge(8,7)
G.add_edge(8,9)
G.add_edge(8,10)

# Layer 4

G.add_edge(11,16)
G.add_edge(11,17)
G.add_edge(12,16)
G.add_edge(12,17)
G.add_edge(13,18)
G.add_edge(13,19)
G.add_edge(14,20)
G.add_edge(14,20)
G.add_edge(15,21)
G.add_edge(15,21)
G.add_edge(5,21)
G.add_edge(5,22)
G.add_edge(7,22)
G.add_edge(7,23)
G.add_edge(9,21)
G.add_edge(9,22)
G.add_edge(10,22)
G.add_edge(10,24)

# Layer 5

G.add_edge(25,26)
G.add_edge(25,27)
G.add_edge(25,28)
G.add_edge(25,29)
G.add_edge(25,30)
G.add_edge(25,31)

nx.draw(G, with_labels=True, node_color="#e377c2")

graph = G

! pip install pennylane

import pennylane as qml
import numpy as np
from pennylane import qaoa as qaoa
from matplotlib import pyplot as plt

cost_h, driver_h = qaoa.max_clique(graph, constrained=False)

print("Cost Hamiltonian")
print(cost_h)
print("Driver Hamiltonian")
print(driver_h)

def build_hamiltonian(graph):
    H = qml.Hamiltonian([], [])

    # Computes the complement of the graph
    graph_c = nx.complement(graph)

    for k in graph_c.nodes:
        # Adds the terms in the first sum
        for edge in graph_c.edges:
            i, j = edge
            if k == i:
                H += 6 * (qml.PauliY(k) @ qml.PauliZ(j) - qml.PauliY(k))
            if k == j:
                H += 6 * (qml.PauliZ(i) @ qml.PauliY(k) - qml.PauliY(k))
        # Adds the terms in the second sum
        H += 6 * qml.PauliY(k)

    return H


print("MaxClique Commutator")
print(build_hamiltonian(graph))

def falqon_layer(beta_k, cost_h, driver_h, delta_t):
    qml.templates.ApproxTimeEvolution(cost_h, delta_t, 1)
    qml.templates.ApproxTimeEvolution(driver_h, delta_t * beta_k, 1)

def build_maxclique_ansatz(cost_h, driver_h, delta_t):
    def ansatz(beta, **kwargs):
        layers = len(beta)
        for w in dev.wires:
            qml.Hadamard(wires=w)
        qml.layer(
            falqon_layer,
            layers,
            beta,
            cost_h=cost_h,
            driver_h=driver_h,
            delta_t=delta_t
        )

    return ansatz

def max_clique_falqon(graph, n, beta_1, delta_t, dev):
    comm_h = build_hamiltonian(graph) # Builds the commutator
    cost_h, driver_h = qaoa.max_clique(graph, constrained=False) # Builds H_c and H_d
    ansatz = build_maxclique_ansatz(cost_h, driver_h, delta_t) # Builds the FALQON ansatz

    beta = [beta_1] # Records each value of beta_k
    energies = [] # Records the value of the cost function at each step

    for i in range(n):
        # Creates a function that can evaluate the expectation value of the commutator
        cost_fn = qml.ExpvalCost(ansatz, comm_h, dev)

        # Creates a function that returns the expectation value of the cost Hamiltonian
        cost_fn_energy = qml.ExpvalCost(ansatz, cost_h, dev)

        # Adds a value of beta to the list and evaluates the cost function
        beta.append(-1 * cost_fn(beta))
        energy = cost_fn_energy(beta)
        energies.append(energy)

    return beta, energies

n = 40
beta_1 = 0.0
delta_t = 0.03

dev = qml.device("default.qubit", wires=graph.nodes) # Creates a device for the simulation
res_beta, res_energies = max_clique_falqon(graph, n, beta_1, delta_t, dev)

plt.plot(range(n+1)[1:], res_energies)
plt.xlabel("Iteration")
plt.ylabel("Cost Function Value")
plt.show()

@qml.qnode(dev)
def prob_circuit():
    build_maxclique_ansatz(cost_h, driver_h, delta_t)(res_beta)
    return qml.probs(wires=dev.wires)

probs = prob_circuit()
plt.bar(range(2**len(dev.wires)), probs)
plt.xlabel("Bit string")
plt.ylabel("Measurement Probability")
plt.show()