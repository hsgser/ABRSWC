# Import libraries
from algorithms import *
from utils import *

# Load data
data_dir    = 'data'
G, pdist = load_data(data_dir)
print(type(G), len(G.nodes()), len(G.edges())) # debug
print(pdist.shape) # debug

# Run experiment
run_time, travel_cost = run_experiment(G, pdist, repeats=1)

# Save experiment results
save_experiment_results(data_dir, 'compare_methods', run_time, travel_cost)