# Import libraries
import os
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
from algorithms import *
from utils import *

# Data path
data_dir    = 'data'
graph_path  = os.path.join(data_dir, 'Shinjuku_processed.graphml')
dist_path   = os.path.join(data_dir, 'pairwise_distance.npy')

# Load network data
G           = nx.read_graphml(graph_path, node_type=int)
print(type(G), len(G.nodes()), len(G.edges())) # debug
V           = list(G.nodes())
N_v         = len(V)

# Load pairwise distance
pdist       = np.load(dist_path)
print(pdist.shape) # debug

# Experiment settings
repeats     = 10    # number of run
z           = 4     # number of seats
epsilon     = 1.5   # extra ratio
S           = 8     # group size
N_u         = 20    # number of users
N_p         = 20    # number of POIs
run_time    = []    # process time
travel_cost = []    # travel cost

for _ in range(repeats):
    # Generate random users, POIs
    rnodes      = np.random.choice(V, size=(N_u+N_p), replace=False)
    U           = rnodes[:N_u]
    P           = rnodes[N_u:]
    print(len(U), len(P)) # debug

    # Find the nearest POI of each node
    N           = find_nearest_POI(V, P, pdist)

    # Greedy approach
    t1          = datetime.now()
    tmp         = greedy_approach(U, N, pdist)
    t2          = datetime.now()
    run_time.append((t2-t1)/timedelta(microseconds=1))
    travel_cost.append(tmp)

    # Optimal solution
    t1          = datetime.now()
    cost, OptCost, OptPOI, J, bestDiv, bestSubgroup = compute_cost(
        U, P, G, z, N, pdist, epsilon)
    OptCost, bestSubgroup = compute_div(U, OptCost, z, N_u)
    t2          = datetime.now()
    run_time.append((t2-t1)/timedelta(microseconds=1))
    travel_cost.append(cal_travel_cost(U, bestSubgroup, OptCost))

    # Approximation solution
    t1          = datetime.now()
    # Step 1
    groups, n_cls   = iterative_nearest_neighbor(U, S, pdist)
    # Step 2
    tmp         = 0
    for i in range(int(n_cls)):
        cost, OptCost, OptPOI, J, bestDiv, bestSubgroup = compute_cost(
            groups[i], P, G, z, N, pdist, epsilon)
        OptCost, bestSubgroup = compute_div(groups[i], OptCost, z, S)
        tmp     += cal_travel_cost(groups[i], bestSubgroup, OptCost)
    t2          = datetime.now()
    run_time.append((t2-t1)/timedelta(microseconds=1))
    travel_cost.append(tmp)

    # Approximation solution without choice of POI
    t1          = datetime.now()
    # Step 1
    groups      = group_by_nearest_POI(U, N)
    # Step 2
    tmp         = 0
    for p in P:
        if len(groups[p]) > 0:
            cost, OptCost, OptPOI, J, bestDiv, bestSubgroup = compute_cost(
                groups[p], P, G, z, N, pdist, epsilon)
            OptCost, bestSubgroup = compute_div(groups[p], OptCost, z, S)
            tmp     += cal_travel_cost(groups[p], bestSubgroup, OptCost)
    t2          = datetime.now()
    run_time.append((t2-t1)/timedelta(microseconds=1))
    travel_cost.append(tmp)

# Save experiment results
save_experiment_results(data_dir, 'compare_methods', run_time, travel_cost)