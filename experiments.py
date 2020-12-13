# Import libraries
from numpy.core.fromnumeric import size
from algorithms import *
from utils import *

# Load data
data_dir    = 'data'
G, pdist = load_data(data_dir)
print(type(G), len(G.nodes()), len(G.edges())) # debug
print(pdist.shape) # debug

# Compare between methods
sizes = np.arange(10)
run_time    = []    
travel_cost = []

for idx in sizes:
    print(f"Run {idx+1}")
    _time, _cost = run_experiment1(G, pdist, repeats=10)
    _time = _time.mean(axis=0)
    _cost = _cost.mean(axis=0)
    run_time.extend(_time)
    travel_cost.extend(_cost)

save_experiment_results(data_dir, 'compare_methods', run_time, travel_cost)

# Compare group size
sizes       = np.arange(4, 18, step=2)
run_time    = []    
travel_cost = [] 

run_time, travel_cost = run_experiment2(G, pdist, param='S', S=sizes, run_optimal=False)
save_experiment_results(data_dir, 'compare_group_size', run_time, travel_cost)

# Compare number of users
sizes       = np.arange(8, 81, step=8)
run_time    = []    
travel_cost = []

for N_u in sizes:
    print(f"N_u = {N_u}")
    _time, _cost = run_experiment1(G, pdist, N_u=N_u, run_optimal=False)
    # Store the mean of all runs
    _time = _time.mean(axis=0)
    _cost = _cost.mean(axis=0)
    run_time.extend(_time)
    travel_cost.extend(_cost)

save_experiment_results(data_dir, 'compare_number_of_user', run_time, travel_cost)

# Compare number of POIs
sizes       = np.arange(10, 101, step=10)
run_time    = []    
travel_cost = []

for N_p in sizes:
    print(f"N_p = {N_p}")
    _time, _cost = run_experiment1(G, pdist, N_p=N_p, run_optimal=False)
    # Store the mean of all runs
    _time = _time.mean(axis=0)
    _cost = _cost.mean(axis=0)
    run_time.extend(_time)
    travel_cost.extend(_cost)

save_experiment_results(data_dir, 'compare_number_of_poi', run_time, travel_cost)

# Compare extra ratio
sizes       = np.linspace(0, 1, num=11)
run_time    = []    
travel_cost = []

run_time, travel_cost = run_experiment2(G, pdist, param='e', epsilon=sizes, run_optimal=False)
save_experiment_results(data_dir, 'compare_extra_ratio', run_time, travel_cost)