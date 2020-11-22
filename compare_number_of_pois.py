# Import libraries
from algorithms import *
from utils import *

# Load data
data_dir    = 'data'
G, pdist = load_data(data_dir)
print(type(G), len(G.nodes()), len(G.edges())) # debug
print(pdist.shape) # debug

# Change number of POIs
sizes       = np.arange(10, 101, step=10)

# Run experiment
run_time    = []    
travel_cost = []    

for N_p in sizes:
    print(f"N_p = {N_p}")
    s_time, s_cost = run_experiment(G, pdist, N_p=N_p, run_optimal=False)
    # Store the mean of all runs
    s_time = s_time.mean(axis=0)
    s_cost = s_cost.mean(axis=0)
    run_time.extend(s_time)
    travel_cost.extend(s_cost)
    
# Save experiment results
save_experiment_results(data_dir, 'compare_number_of_poi', run_time, travel_cost)