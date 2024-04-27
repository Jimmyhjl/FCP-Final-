import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
	def DFS(self, root_node:Node): # returns the best route to reach every node from the root node
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
			routes = self.DFS(node)
			for key in routes.keys():
				if key == node: # dont count the route to itself in the calculation
					continue
				length_sum += len(routes[key])
			average_length_sum += length_sum/(len(routes.keys())-1)
		return average_length_sum/len(self.nodes)
				

	def make_random_network(self, N, connection_probability):
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

	def make_ring_network(self, N, neighbour_range=1):
		pass
		#Your code  for task 4 goes here

	def make_small_world_network(self, N, re_wire_prob=0.2):
		pass
		#Your code for task 4 goes here

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

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
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
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

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
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

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
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''

	#Your code for task 1 goes here

	return np.random * population

def ising_step(population, external=0.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)

	agreement = calculate_agreement(population, row, col, external=0.0)

	if agreement < 0:
		population[row, col] *= -1

	#Your code for task 1 goes here

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
	assert(calculate_agreement(population,1,1)==4), "Test 1"

	population[1, 1] = 1.
	assert(calculate_agreement(population,1,1)==-4), "Test 2"

	population[0, 1] = 1.
	assert(calculate_agreement(population,1,1)==-2), "Test 3"

	population[1, 0] = 1.
	assert(calculate_agreement(population,1,1)==0), "Test 4"

	population[2, 1] = 1.
	assert(calculate_agreement(population,1,1)==2), "Test 5"

	population[1, 2] = 1.
	assert(calculate_agreement(population,1,1)==4), "Test 6"

	"Testing external pull"
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1,1)==3), "Test 7"
	assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
	assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
	assert(calculate_agreement(population,1,1,-10)==14), "Test 10"

	print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_axis_off()
	im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

	# Iterating an update 100 times
	for frame in range(100):
		# Iterating single steps 1000 times to form an update
		for step in range(1000):
			ising_step(population, external)
		print('Step:', frame, end='\r')
		plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main():
    pass
	#Your code for task 2 goes here

def test_defuant():
    pass
	#Your code for task 2 goes here


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
    pass
	#You should write some code for handling flags here

if __name__=="__main__":
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)
	
	for node in network.nodes:
		assert(len(node.get_neighbour_indexes())==num_nodes-1), node.get_neighbour_indexes()
	print("Pass")
	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	print("Pass")
	assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
	print("Pass")
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()
	print("Pass")
