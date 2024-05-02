import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

	def get_neighbour_indexes(self):
		neighbour_indexes = []
		for (i, value) in enumerate(self.connections):
			if value == 0:
				continue
			neighbour_indexes.append(i)
		return neighbour_indexes

class Network: 
	def __init__(self, nodes=None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 
	
	'''
	Mean Degree. degree is the number of edges of each node, find average
	
	Mean path length, breadth-first-search to find lengths, find average of all nodes
 
	Mean clustering co-efficient, the clustering coefficient measures the fraction of a nodes neighbours that connect to eachother. 
	number of possible connections for n neighbors is S = n*(n-1)/2. then find x (number of connections between neighbors) and then find x/S
	'''
	def BFS(self, root_node:Node): # returns the best route to reach every node from the root node
		visited = [root_node] # track which nodes have already been visited
		queue = [root_node] # list of unvisited nodes to visit (remove from front, to act as a queue)
		routes = {root_node: []} # dictionary of each node, and a list of the least amount of nodes to reach them
  
		while queue:
			current_node = queue.pop(0)
			neighbour_indexes = current_node.get_neighbour_indexes()
			for i in neighbour_indexes:
				# get the node object of the neighbour
				neighbour = self.nodes[i]
				if neighbour in visited: # if neighbour has been visited. Skip it
					continue
				# neighbour hasn't been visited, add it to the queue and add it to the visited list so that its route wont be re-established
				visited.append(neighbour)
				queue.append(neighbour)
				# establish and save new route
				route = routes[current_node].copy()
				route.append(current_node)
				routes[neighbour] = route
		return routes
  
	def get_mean_degree(self):
		sum_of_edges = 0
		for node in self.nodes:
			sum_of_edges += len(node.get_neighbour_indexes())
		return sum_of_edges/len(self.nodes)

	def get_mean_clustering(self):
		coefficient_sum = 0
		for node in self.nodes:
			neighbour_indexes = node.get_neighbour_indexes()
			number_of_neighbours = len(neighbour_indexes)
			maximum_clustering = number_of_neighbours*(number_of_neighbours-1)/2
			if maximum_clustering == 0:
				continue

			checked_edges = []
			x = 0 # track how many neighbours are connected to eachother
			for neighbour_index in neighbour_indexes:
				neighbour = self.nodes[neighbour_index]
				indexes = neighbour.get_neighbour_indexes()
				for i in indexes:
					if i == self.nodes.index(node): # check if the connection is back to the original node
						continue
					if {neighbour_index, i} in checked_edges: # check if the edge has already been checked
						continue
					checked_edges.append({neighbour_index, i}) # usage of a set so that the order which the edge is checked is irrelevant
					if i in neighbour_indexes: # edge is valid therefore count it
						x += 1
			coefficient_sum += x/maximum_clustering
		return coefficient_sum/len(self.nodes)

	def get_mean_path_length(self):
		average_length_sum = 0
		for node in self.nodes:
			length_sum = 0
			routes = self.BFS(node)
			for key in routes.keys():
				if key == node: # dont count the route to itself in the calculation
					continue
				length_sum += len(routes[key])
			if length_sum != 0:
				average_length_sum += length_sum/(len(routes.keys())-1)
		return average_length_sum/len(self.nodes)
				

	def make_random_network(self, N, connection_probability=0.25):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			# connections is defined as a list of length equal to the number of nodes, where there is a 0 or 1 depending on if there is an edge between them
			# identifying which nodes are connected to which requires tracking their index in the node list (stored in the network)
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
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
		self.make_ring_network(N)
		for i in range(0, N):
			for j in range(i + 1, N):
				if self.nodes[i].connections[j] == 1:
					if np.random.rand() < re_wire_prob:
						connected_nodes = np.where(self.nodes[i].connections == 1)[0] # find indexes of neighbours
						possible_nodes = np.setdiff1d(np.arange(N), connected_nodes) # exclude neighbours from the list of possible nodes to connect to
						possible_nodes = possible_nodes[possible_nodes != i] # prevent reconnecting to itself

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
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value/2))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():
	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_mean_clustering()==0), network.get_mean_clustering()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==5), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()

	print("All tests passed")


'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''
def create_array(rows=100, columns=100, probability=0.5):
    """
    Function to create a numpy array with specified number of rows and columns,
    initialized with values -1 and 1 according to the given probability.

    Args:
    - rows (int): Number of rows in the array. Default is 100.
    - columns (int): Number of columns in the array. Default is 100.
    - probability (float): Probability of getting value 1. Default is 0.5.

    Returns:
    - array (numpy.ndarray): Numpy array of shape (rows, columns) with values -1 and 1.
    """
    array = np.random.choice([-1, 1], size=(rows, columns), p=[1 - probability, probability])
    return array

def calculate_agreement(population, row, col, external=0):
    """
    Function to calculate the agreement for a specific cell in an Ising model.

    Args:
    - population (numpy.ndarray): 2D numpy array representing the Ising model.
    - row (int): Row index of the cell.
    - col (int): Column index of the cell.
    - external (float): External influence on the cell's agreement. Default is 0.

    Returns:
    - Di (float): Agreement value for the cell at (row, col).
    """
    center = population[row, col]
    left_neighbor = population[row, col - 1]
    right_neighbor = population[row, (col + 1) % population.shape[1]]
    upper_neighbor = population[row - 1, col]
    bottom_neighbor = population[(row + 1) % population.shape[0], col]

    PO = (left_neighbor + right_neighbor + upper_neighbor + bottom_neighbor + external) * center
    return PO


def ising_step(population, alpha, external=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            alpha (float) - tolerance parameter
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external)

    if agreement < 0:
        population[row, col] *= -1
    else:
        probability_of_flip = np.exp(-agreement / alpha)
        if np.random.rand() < probability_of_flip:
            population[row, col] *= -1



def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''
    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)


def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''
	print("Testing ising model calculations")
	population = -np.ones((3, 3))
	assert (calculate_agreement(population, 1, 1) == 4), "Test 1"

	population[1, 1] = 1.
	assert (calculate_agreement(population, 1, 1) == -4), "Test 2"

	population[0, 1] = 1.
	assert (calculate_agreement(population, 1, 1) == -2), "Test 3"

	population[1, 0] = 1.
	assert (calculate_agreement(population, 1, 1) == 0), "Test 4"

	population[2, 1] = 1.
	assert (calculate_agreement(population, 1, 1) == 2), "Test 5"

	population[1, 2] = 1.
	assert (calculate_agreement(population, 1, 1) == 4), "Test 6"

	"Testing external pull"
	population = -np.ones((3, 3))
	assert (calculate_agreement(population, 1, 1, 1) == 3), "Test 7"
	assert (calculate_agreement(population, 1, 1, -1) == 5), "Test 8"
	assert (calculate_agreement(population, 1, 1, 10) == -6), "Test 9"
	assert (calculate_agreement(population, 1, 1, -10) == 14), "Test 10"

	print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
	fig = plt.figure()
	plt.suptitle(f"External: {external}, alpha: {alpha}", fontsize=16)
	ax = fig.add_subplot(111)
	ax.set_axis_off()
	im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

	# Iterating an update 100 times
	for frame in range(100):

		# Iterating single steps 1000 times to form an update
		for _ in range(1000):
			ising_step(population, external)
		print('Step:', frame, end='\r')
		plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''


def defuant_main(opinions, beta=0.2, threshold=0.2):
	epoch = 100
	each_epoch_iters = 1000
	for _ in range(epoch):
		for _ in range(each_epoch_iters):
			opinions = defuant_iter(opinions, beta, threshold)
	return opinions

def defuant_iter(opinions, beta=0.2, threshold=0.2):
	n = len(opinions)
	i, j = np.random.choice(n, 2, replace=False)
	if abs(opinions[i] - opinions[j]) < threshold:
		opinions[i] += beta * (opinions[j] - opinions[i])
		opinions[j] += beta * (opinions[i] - opinions[j])
	return opinions

def test_defuant(opinions, beta=0.2, threshold=0.2):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
	plt.suptitle(f"Coupling: {beta}, Threshold: {threshold}", fontsize=16)
	ax1.set_xticks(np.arange(0, 1.1, 0.1))
	ax1.set_xlabel('Opinion')
	ax1.set_ylabel('Count')
	ax1.set_title('fig1')
	ax2.plot()
	ax2.set_title('fig2')
	x_values = []
	y_values = []
	epoch = 100
	each_epoch_iters = 1000
	for frame in range(epoch):
		for step in range(each_epoch_iters):
			opinions = defuant_iter(opinions, beta, threshold)
		bin_size = 0.05
		bins = np.arange(0, 1 + bin_size, bin_size)
		hist, bin_edges = np.histogram(opinions, bins=bins)
		ax1.bar(bin_edges[:-1], hist, width=bin_size, edgecolor='black', align='edge')

		ax1.cla()
		ax1.bar(bin_edges[:-1], hist, width=bin_size, edgecolor='black', align='edge', color='blue')
		ax1.set_xticks(np.arange(0, 1.1, 0.1))
		ax1.set_xlabel('Opinion')
		ax1.set_ylabel('Count')
		ax1.set_title('Opinion Distribution - Frame {}'.format(frame+1))

		ax2.cla()
		x_values.extend([frame] * len(opinions))
		y_values.extend(opinions)
		ax2.scatter(x_values, y_values, alpha=0.6, color='red')
		ax2.set_xlim(0, epoch)
		ax2.set_ylim(0, 1)
		ax2.set_xlabel('Iteration')
		ax2.set_ylabel('Opinion values')
		ax2.set_title('Opinion Trace Over Iterations')

		plt.pause(0.1)
	plt.show()
	return opinions

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def get_args():
    
	parser = argparse.ArgumentParser(description="Produce and visualize social connections.")

	# Optional Arguments
	parser.add_argument('-ising_model', action='store_true')
	parser.add_argument('-test_ising', action='store_true')

	parser.add_argument('-defuant', action='store_true')
	parser.add_argument('-test_defuant', action='store_true')

	parser.add_argument('-test_networks', action='store_true')

	# Optional positional arguments

	## for networks
	parser.add_argument('-network', type=int)
	parser.add_argument('-connection_probability', type=float, default=0.25)
	parser.add_argument('-ring_network', type=int)
	parser.add_argument('-small_world', type=int)
	parser.add_argument('-re_wire', default=0.2, type=float)

	## for ising
	parser.add_argument('-external', default=0, type=float)
	parser.add_argument('-alpha', default=1, type=float)

	## for defuant
	parser.add_argument('-beta', default=0.2, type=float)
	parser.add_argument('-threshold', default=0.2, type=float)

	args = parser.parse_args()
	return args

def main():
	args = get_args()
 
	if args.ising_model:
		print("ising model")
		print(f"external = {args.external} alpha = {args.alpha}")
		population = create_array()
		ising_main(population, args.alpha, args.external)
			
	if args.test_ising:
		print("testing ising model")
		test_ising()

	if args.defuant:
		print("defuant model")
		print(f"beta = {args.beta} threshold = {args.threshold}")
		opinions = np.random.rand(100)
		opinions = defuant_main(opinions, args.beta, args.threshold)
		
	if args.test_defuant:
		print("testing defuant model")
		opinions = np.random.rand(100)
		test_defuant(opinions, args.beta, args.threshold)
  
	if args.network:
		print(f"random network n = {args.network}")
		network = Network()
		network.make_random_network(args.network, args.connection_probability)
		print(f"Mean degree: {network.get_mean_degree()}")
		print(f"Average path length: {network.get_mean_path_length()}")
		print(f"Clustering co-efficient: {network.get_mean_clustering()}")
		network.plot()
		plt.show()
  
	if args.test_networks:
		print("testing networks")
		test_networks()

	if args.ring_network:
		print(f"ring network n = {args.ring_network}")
		network = Network()
		network.make_ring_network(args.ring_network)
		network.plot()
		plt.show()

	if args.small_world:
		print(f"small world n = {args.small_world} re_wire = {args.re_wire}")
		network = Network()
		network.make_small_world_network(args.small_world, args.re_wire)
		network.plot()
		plt.show()


if __name__ == "__main__":
    main()