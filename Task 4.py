import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse


class Node:

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value


class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def make_random_network(self, N, connection_probability=0.5):
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=2):
        neighbouring = np.zeros((N, N), dtype=int)
        for i in range(0, N):

            # Set connections to left and right neighbours
            for j in range(1, neighbour_range + 1):

                # Set the connection to the left neighbouring node
                neighbouring[i, (i - j) % N] = 1

                # Set the connection to the right neighbouring node
                neighbouring[i, (i + j) % N] = 1

        # Create nodes with random values and connections
        self.nodes = [Node(np.random.rand(), i, neighbouring[i]) for i in range(N)]

    def make_small_world_network(self, N, re_wire_prob=0.2):
        n = len(self.nodes)
        for i in range(0, n):
            for j in range(i + 1, n):
                if self.nodes[i].connections[j] == 1:
                    if np.random.rand() < re_wire_prob:
                        connected_nodes = np.where(self.nodes[i].connections == 1)[0]
                        possible_nodes = np.setdiff1d(np.arange(n), connected_nodes)
                        possible_nodes = possible_nodes[possible_nodes != i]

                        if len(possible_nodes) > 0:
                            # Select a random node from the possible nodes
                            new_node = np.random.choice(possible_nodes)

                            # Break the original connection
                            self.nodes[i].connections[j] = 0
                            self.nodes[j].connections[i] = 0

                            # Establish a new connection
                            self.nodes[i].connections[new_node] = 1
                            self.nodes[new_node].connections[i] = 1

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

        plt.show()


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Produce and visualize small-world networks.")

    # Define command line arguments for number of nodes
    parser.add_argument('-n', '--nodes', default=10, type=int)

    # Define command line arguments for the probability re-wiring
    parser.add_argument('-p', '--probability', default=0.2, type=float)
    args = parser.parse_args()

    # Make a Network object to generate small-world networks
    network = Network()

    # Generate and plot a ring network with the specified number of nodes
    network.make_ring_network(args.nodes)
    network.plot()

    # Generate and plot a small-world network with the specified number of nodes and re-wiring probability
    network.make_small_world_network(args.nodes, args.probability)
    network.plot()


if __name__ == "__main__":
    main()
