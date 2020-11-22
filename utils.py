import os
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
from algorithms import *

def load_data(data_dir):
    """
    An utility function to load data.

    Parameters
    ----------
    data_dir: path-like object
        Path of data folder.
    
    Returns
    -------
    G: networkx graph
        Road network.
    dist: array
        Pairwise distance.
    """
    graph_path  = os.path.join(data_dir, 'Shinjuku_processed.graphml')
    dist_path   = os.path.join(data_dir, 'pairwise_distance.npy')

    # Load network data
    G           = nx.read_graphml(graph_path, node_type=int)
    # Load pairwise distance
    dist       = np.load(dist_path)

    return G, dist

def save_experiment_results(data_dir, prefix, run_time, travel_cost):
    """
    An utility function to save experiment results with time info.

    Parameters
    ----------
    data_dir: path-like object
        Path of data folder.
    prefix: str
        Name of the experiment.
    run_time: array
        List of processing time.
    travel_cost: array
        List of travel cost.
    """
    time_dir    = os.path.join(data_dir, 'time')
    cost_dir    = os.path.join(data_dir, 'cost')
    curr_time   = datetime.now()
    postfix     = '{0}_{1}_{2}_{3}_{4}_{5}'.format(
        curr_time.year, curr_time.month, curr_time.day, 
        curr_time.hour, curr_time.minute, curr_time.second)
    filename    = prefix + '_' + postfix + '.npy'
    time_path   = os.path.join(time_dir, filename)
    cost_path   = os.path.join(cost_dir, filename)
    with open(time_path, 'wb') as f:
        np.save(f, run_time)
    with open(cost_path, 'wb') as f:
        np.save(f, travel_cost)
    
def run_experiment(G, pdist, repeats=10, z=4, epsilon=1.5, S=8, N_u=16, N_p=20, run_optimal=True):
    """
    An utility function to run experiment.

    Parameters
    ----------
    G: networkx graph
        Road network.
    pdist: array
        Pairwise distance.
    repeats: int, default = 10
        Number of run.
    z: int, default = 4
        Number of seats.
    epsilon: float, default = 1.5
        Extra ratio.
    S: int, default = 8
        Group size.
    N_u: int, default = 16
        Number of users.
    N_p: int, default = 20
        Number of POIs.
    run_optimal: bool, default=True
        If true, run the optimal solution. Otherwise, do not run the optimal solution.
    
    Returns
    -------
    run_time: array
        Processing time.
    travel_cost: array
        Total travel distance.
    """
    V           = list(G.nodes())
    run_time    = []    # process time
    travel_cost = []    # travel cost

    for _ in range(repeats):
        # Generate random users, POIs
        rnodes      = np.random.choice(V, size=(N_u+N_p), replace=False)
        U           = rnodes[:N_u]
        P           = rnodes[N_u:]

        # Find the nearest POI of each node
        N           = find_nearest_POI(V, P, pdist)

        # Greedy approach
        t1          = datetime.now()
        tmp         = greedy_approach(U, N, pdist)
        t2          = datetime.now()
        run_time.append((t2-t1)/timedelta(microseconds=1))
        travel_cost.append(tmp)

        # Optimal solution
        if run_optimal:
            t1          = datetime.now()
            _, OptCost, _, _, _, bestSubgroup = compute_cost(
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
            _, OptCost, _, _, _, bestSubgroup = compute_cost(
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
                _, OptCost, _, _, _, bestSubgroup = compute_cost(
                    groups[p], P, G, z, N, pdist, epsilon)
                OptCost, bestSubgroup = compute_div(groups[p], OptCost, z, S)
                tmp     += cal_travel_cost(groups[p], bestSubgroup, OptCost)
        t2          = datetime.now()
        run_time.append((t2-t1)/timedelta(microseconds=1))
        travel_cost.append(tmp)
    
    run_time    = np.array(run_time).reshape(repeats, -1)
    travel_cost = np.array(travel_cost).reshape(repeats, -1)

    return run_time, travel_cost