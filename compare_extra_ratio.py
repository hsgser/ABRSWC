# Import libraries
from algorithms import *
from utils import *

# Load data
data_dir    = 'data'
G, pdist = load_data(data_dir)
print(type(G), len(G.nodes()), len(G.edges())) # debug
print(pdist.shape) # debug

# Change extra ratio
sizes       = np.linspace(1, 2, num=11)

# Run experiment
run_time    = []    
travel_cost = []    

for e in sizes:
    print(f"Epsilon = {e}")
    s_time, s_cost = run_experiment(G, pdist, epsilon=e, run_optimal=False)
    # Store the mean of all runs
    s_time = s_time.mean(axis=0)
    s_cost = s_cost.mean(axis=0)
    run_time.extend(s_time)
    travel_cost.extend(s_cost)
    
# Save experiment results
save_experiment_results(data_dir, 'compare_extra_ratio', run_time, travel_cost)