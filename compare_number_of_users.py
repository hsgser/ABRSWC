# Import libraries
from algorithms import *
from utils import *

# Load data
data_dir    = 'data'
G, pdist = load_data(data_dir)
print(type(G), len(G.nodes()), len(G.edges())) # debug
print(pdist.shape) # debug

# Change number of users
sizes       = np.arange(4, 101, step=4)

# Run experiment
run_time    = []    
travel_cost = []    

for N_u in sizes:
    print(f"N_u = {N_u}")
    s_time, s_cost = run_experiment(G, pdist, N_u=N_u, run_optimal=False)
    # Store the mean of all runs
    s_time = s_time.mean(axis=0)
    s_cost = s_cost.mean(axis=0)
    run_time.extend(s_time)
    travel_cost.extend(s_cost)
    
# Save experiment results
save_experiment_results(data_dir, 'compare_number_of_user', run_time, travel_cost)