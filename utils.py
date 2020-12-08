import os
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
from algorithms import *
import matplotlib.pyplot as plt

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
    
def run_experiment1(G, pdist, repeats=10, z=4, N_u=16, N_p=20, run_optimal=True):
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
    N_u: int, default = 16
        Number of users.
    N_p: int, default = 20
        Number of POIs.
    run_optimal: bool, default = True
        If true, run the optimal solution. Otherwise, do not run the optimal solution.
    
    Returns
    -------
    run_time: array
        Processing time.
    travel_cost: array
        Total travel distance.
    """
    # Settings
    epsilon     = 1.5
    S           = 8
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
            OptCost, bestSubgroup = compute_div(U, OptCost, z)
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
            OptCost, bestSubgroup = compute_div(groups[i], OptCost, z)
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
                OptCost, bestSubgroup = compute_div(groups[p], OptCost, z)
                tmp     += cal_travel_cost(groups[p], bestSubgroup, OptCost)
        t2          = datetime.now()
        run_time.append((t2-t1)/timedelta(microseconds=1))
        travel_cost.append(tmp)
    
    run_time    = np.array(run_time).reshape(repeats, -1)
    travel_cost = np.array(travel_cost).reshape(repeats, -1)

    return run_time, travel_cost

def run_experiment2(G, pdist, param, repeats=10, z=4, epsilon=1.5, S=8, run_optimal=True):
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
    param: string, {'S', 'e'}
        Name of changing parameters.
        If 'S', changes group size. S must be array-like (list, tuple, np.ndarray).
        If 'e', changes epsilon. epsilon must be array-like (list, tuple, np.ndarray).
    z: int, default = 4
        Number of seats.
    epsilon: float or array-like, default = 1.5
        Extra ratio.
    S: int or array-like, default = 8
        Group size.
    run_optimal: bool, default = True
        If true, run the optimal solution. Otherwise, do not run the optimal solution.
    
    Returns
    -------
    run_time: array
        Processing time.
    travel_cost: array
        Total travel distance.
    """
    # Parameters check
    if param == 'S':
        assert isinstance(S, (list, tuple, np.ndarray))
        epsilon = [epsilon for _ in range(len(S))]
    elif param == 'e':
        assert isinstance(epsilon, (list, tuple, np.ndarray))
        S       = [S for _ in range(len(epsilon))]
    else:
        raise ValueError("param can only be 'S' or 'e'.")

    # Settings
    N_u         = 16
    N_p         = 20
    V           = list(G.nodes())
    run_time    = []    # process time
    travel_cost = []    # travel cost

    for idx in range(repeats):
        print(f"Run {idx}")
        # Generate random users, POIs
        rnodes      = np.random.choice(V, size=(N_u+N_p), replace=False)
        U           = rnodes[:N_u]
        P           = rnodes[N_u:]

        # Find the nearest POI of each node
        N           = find_nearest_POI(V, P, pdist)

        _time = []
        _cost = []

        for _S, _epsilon in zip(S, epsilon):
            # Greedy approach
            t1          = datetime.now()
            tmp         = greedy_approach(U, N, pdist)
            t2          = datetime.now()
            _time.append((t2-t1)/timedelta(microseconds=1))
            _cost.append(tmp)

            # Optimal solution
            if run_optimal:
                t1          = datetime.now()
                _, OptCost, _, _, _, bestSubgroup = compute_cost(
                    U, P, G, z, N, pdist, _epsilon)
                OptCost, bestSubgroup = compute_div(U, OptCost, z)
                t2          = datetime.now()
                _time.append((t2-t1)/timedelta(microseconds=1))
                _cost.append(cal_travel_cost(U, bestSubgroup, OptCost))

            # Approximation solution
            t1          = datetime.now()
            # Step 1
            groups, n_cls   = iterative_nearest_neighbor(U, _S, pdist)
            # Step 2
            tmp         = 0
            for i in range(int(n_cls)):
                _, OptCost, _, _, _, bestSubgroup = compute_cost(
                    groups[i], P, G, z, N, pdist, _epsilon)
                OptCost, bestSubgroup = compute_div(groups[i], OptCost, z)
                tmp     += cal_travel_cost(groups[i], bestSubgroup, OptCost)
            t2          = datetime.now()
            _time.append((t2-t1)/timedelta(microseconds=1))
            _cost.append(tmp)

            # Approximation solution without choice of POI
            t1          = datetime.now()
            # Step 1
            groups      = group_by_nearest_POI(U, N)
            # Step 2
            tmp         = 0
            for p in P:
                if len(groups[p]) > 0:
                    _, OptCost, _, _, _, bestSubgroup = compute_cost(
                        groups[p], P, G, z, N, pdist, _epsilon)
                    OptCost, bestSubgroup = compute_div(groups[p], OptCost, z)
                    tmp     += cal_travel_cost(groups[p], bestSubgroup, OptCost)
            t2          = datetime.now()
            _time.append((t2-t1)/timedelta(microseconds=1))
            _cost.append(tmp)
        
        run_time.append(_time)
        travel_cost.append(_cost)
    
    run_time    = np.array(run_time).mean(axis=0)
    travel_cost = np.array(travel_cost).mean(axis=0)

    return run_time, travel_cost

def plot_proc_time(df, X, exp_name, ylim, fig_dir='data/figure', xlabel='Run', plot_optimal=False, legend_loc=(0, 0.5)):
    """
    An utility function to plot the mean processing time.

    Parameters
    ----------
    df: DataFrame
        A DataFrame contains the mean processing time.
    X: array
        Horizontal coordinates of data.
    exp_name: string
        Name of the experiment.
    ylim: array, len = 2
        y-axis limits.
    fig_dir: path-like object, default = 'data/figure'
        Path of figure folder.
    xlabel: string, default = 'Run'
        x-axis label.
    plot_optimal: bool, default = False
        If true, plot the optimal solution. Otherwise, do not plot the optimal solution.
    legend_loc: array, len = 2, default = (0, 0.5)
        The location of the legend
    """
    _, ax = plt.subplots(figsize=(10, 7))
    if plot_optimal:
        ax.plot(X, df['optimal'], marker='o', markersize=8, color='blue', label='optimal')
    ax.plot(X, df['approx'], marker='d',  markersize=8, color='orange', label='approx')
    ax.plot(X, df['approx_woc'], marker='s',  markersize=8, color='green', label='approx_woc')
    ax.set_yscale('log')
    ax.set_ylim(*ylim)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel('Mean processing time [s]', fontsize=30)
    ax.grid(True, axis='y')
    ax.spines['bottom'].set_alpha(0.4)
    ax.spines['top'].set_alpha(0.4)
    ax.spines['left'].set_alpha(0.4)
    ax.spines['right'].set_alpha(0.4)
    ax.legend(loc=legend_loc, framealpha=0.0, fontsize=20)
    plt.savefig(os.path.join(fig_dir, f'{exp_name}_proc_time.png'), bbox_inches='tight')
    plt.show()

def plot_travel_cost(df, X, exp_name, ylim, fig_dir='data/figure', xlabel='Run', plot_optimal=False, legend_loc=(0.5, 0.69)):
    """
    An utility function to plot the total travel cost.

    Parameters
    ----------
    df: DataFrame
        A DataFrame contains the total travel cost.
    X: array
        Horizontal coordinates of data.
    exp_name: string
        Name of the experiment.
    ylim: array, len = 2
        y-axis limits.
    fig_dir: path-like object, default = 'data/figure'
        Path of figure folder.
    xlabel: string, default = 'Run'
        x-axis label.
    plot_optimal: bool, default = False
        If true, plot the optimal solution. Otherwise, do not plot the optimal solution.
    legend_loc: array, len = 2, default = (0.5, 0.69)
        The location of the legend
    """
    _, ax = plt.subplots(figsize=(10, 7))
    if plot_optimal:
        ax.plot(X, df['optimal'], marker='o', markersize=8, color='blue', label='optimal')
    ax.plot(X, df['approx'], marker='d', markersize=8, color='orange', label='approx')
    ax.plot(X, df['approx_woc'], marker='s', markersize=8, color='green', label='approx_woc')
    ax.plot(X, df['greedy'], marker='v', markersize=8, color='red', label='greedy')
    ax.set_ylim(*ylim)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel('Total travel cost [m]', fontsize=30)
    ax.grid(True, axis='y')
    ax.spines['bottom'].set_alpha(0.4)
    ax.spines['top'].set_alpha(0.4)
    ax.spines['left'].set_alpha(0.4)
    ax.spines['right'].set_alpha(0.4)
    ax.legend(loc=legend_loc, framealpha=0.0, fontsize=20)
    plt.savefig(os.path.join(fig_dir, f'{exp_name}_travel_cost.png'), bbox_inches='tight')
    plt.show()

def plot_approx_ratio(df, X, exp_name, ylim, fig_dir='data/figure', xlabel='Run', legend_loc=(0.3, 0.63)):
    """
    An utility function to plot the approximation ratio.

    Parameters
    ----------
    df: DataFrame
        A DataFrame contains the approximation ratio.
    X: array
        Horizontal coordinates of data.
    exp_name: string
        Name of the experiment.
    ylim: array, len = 2
        y-axis limits.
    fig_dir: path-like object, default = 'data/figure'
        Path of figure folder.
    xlabel: string, default = 'Run'
        x-axis label.
    plot_optimal: bool, default = False
        If true, plot the optimal solution. Otherwise, do not plot the optimal solution.
    legend_loc: array, len = 2, default = (0.3, 0.63)
        The location of the legend
    """
    _, ax = plt.subplots(figsize=(10, 7))
    ax.plot(X, df['approx_ratio'], marker='d', markersize=8, color='orange', label='approx')
    ax.plot(X, df['approx_woc_ratio'], marker='s', markersize=8, color='green', label='approx_woc')
    ax.plot(X, df['greedy_ratio'],  marker='v', markersize=8, color='red', label='greedy')
    ax.set_ylim(*ylim)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel('Approximation ratio', fontsize=30)
    ax.grid(True, axis='y')
    ax.spines['bottom'].set_alpha(0.4)
    ax.spines['top'].set_alpha(0.4)
    ax.spines['left'].set_alpha(0.4)
    ax.spines['right'].set_alpha(0.4)
    ax.legend(loc=legend_loc, framealpha=0.0, fontsize=20)
    plt.savefig(os.path.join(fig_dir, f'{exp_name}_approx_ratio.png'), bbox_inches='tight')
    plt.show()

def plot_rs_efficiency(df, X, exp_name, ylim, fig_dir='data/figure', xlabel='Run', plot_optimal=False, legend_loc=(0.3, 0.63)):
    """
    An utility function to plot the ride-sharing efficiency.

    Parameters
    ----------
    df: DataFrame
        A DataFrame contains the ride-sharing efficiency.
    X: array
        Horizontal coordinates of data.
    exp_name: string
        Name of the experiment.
    ylim: array, len = 2
        y-axis limits.
    fig_dir: path-like object, default = 'data/figure'
        Path of figure folder.
    xlabel: string, default = 'Run'
        x-axis label.
    plot_optimal: bool, default = False
        If true, plot the optimal solution. Otherwise, do not plot the optimal solution.
    legend_loc: array, len = 2, default = (0.5, 0.63)
        The location of the legend
    """
    _, ax = plt.subplots(figsize=(10, 7))
    if plot_optimal:
        ax.plot(X, df['optimal_gain'], marker='o', markersize=8, color='blue', label='optimal')
    ax.plot(X, df['approx_gain'], marker='d', markersize=8, color='orange', label='approx')
    ax.plot(X, df['approx_woc_gain'], marker='s', markersize=8, color='green', label='approx_woc')
    ax.set_ylim(*ylim)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel('Ride-sharing efficiency', fontsize=30)
    ax.grid(True, axis='y')
    ax.spines['bottom'].set_alpha(0.4)
    ax.spines['top'].set_alpha(0.4)
    ax.spines['left'].set_alpha(0.4)
    ax.spines['right'].set_alpha(0.4)
    ax.legend(loc=legend_loc, framealpha=0.0, fontsize=20)
    plt.savefig(os.path.join(fig_dir, f'{exp_name}_rs_efficiency.png'), bbox_inches='tight')
    plt.show()