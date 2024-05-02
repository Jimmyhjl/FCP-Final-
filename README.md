Task 1

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

Task 4
Clone the repository or download the small_world_network.py file.
Navigate to the directory containing small_world_network.py.
Run the script using Python:
python small_world_network.py [-h] [-nodes NODES] [-re_wire RE_WIRE]
Optional arguments:
-nodes NODES: Number of nodes in the network (default: 10).
-re_wire RE_WIRE: Probability of re-wiring edges in the small-world network (default: 0.98).
