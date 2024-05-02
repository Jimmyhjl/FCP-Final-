# Quickstart

The code is operated using CLI flags when upon execution.

**List of arguments and their purpose**

* -ising_model
  * -external
  * -alpha
  * -H
* -test_ising | executes the test_ising function to validate the code
* -defuant
  * -beta
  * -threshold
* -test_defuant | executes the test_defuant function to validate the code
* -test_networks | executes the test_networks function to validate the code
* -network | takes an integer value for the number of nodes to plot a random network
  * connection_probability | optionally specify the chance a node will connect to a neighbour (float) default is 0.25
* -ring_network | takes an integer value for the number of nodes
* -small_world | takes an integer value for the number of nodes
  * -re_wire | optionally specifiy the rewire probability (float) default is 0.2

**Example usage:**
python3 .\\FCP_assignment_.py -ising_model -external 0.1 -alpha 0.5

# Task Specific

## Task 1

Creating the Ising Model (create_array Function):
Generates a numpy array representing the Ising model with specified dimensions.
Initializes the array with spin values (-1 or 1) based on a given probability.
Calculating Agreement (calculate_agreement Function):
Computes the agreement for a specific cell in the Ising model.
Considers the spin of the cell and its neighboring spins to determine agreement.
Updating the Model (ising_step Function):
Performs a single update of the Ising model by flipping spins based on the calculated agreement.
Randomly selects a cell and evaluates whether to flip its spin.
Flipping depends on the agreement and a tolerance parameter (alpha).
Visualization (plot_ising Function):
Displays a plot of the Ising model using Matplotlib.
Converts the spin values in the array to colors for visualization.
Main Simulation (ising_main Function):
Initializes the plot for visualization.
Iterates through updates of the Ising model, updating the plot to show the model's evolution over time.
Each update consists of multiple steps, adjusting the spins according to the rules of the Ising model.
Customization:
Parameters such as lattice size, initial spin probability, tolerance (alpha), and external influence can be adjusted to customize the simulation.
Testing:
Includes a testing function (test_ising) to verify the correctness of agreement calculations under different scenarios.

## Task 2

\-defuant: Run the Deffuant model simulation.
\-beta: Specify the beta parameter for the Deffuant model (default: 0.2).
\-threshold: Specify the threshold parameter for the Deffuant model (default: 0.2).
\-test_defuant: Run tests for the Deffuant model.
\-use_network: Use network structure for the simulation.

## Task 3

Functions which were not specified for the task were added to the code.

* get_neighbour_indexes() defined in the Node class, returns the network's indices for every neighbour of a node
* BFS(root_node) defined in the Network class, peforms a Breadth-first-search to find the shortest route to every other node in the network from the root_node

test the validitiy of the code by executing
python3 .\\FCP_assignment_.py -network 10
python3 .\\FCP_assignment_.py -test_networks

## Task 4

install matplotlib and numpy
Clone the repository or download the small_world_network.py file.
Navigate to the directory containing small_world_network.py.
Run the script using Python:
python small_world_network.py \[-h\] \[-nodes NODES\] \[-re_wire RE_WIRE\]
Optional arguments:
\-nodes NODES: Number of nodes in the network (default: 10).
\-re_wire RE_WIRE: Probability of re-wiring edges in the small-world network (default: 0.98).