# Import libraries
import numpy as np
from heapq import heappush, heappop
from collections import defaultdict
from itertools import combinations, chain

def find_nearest_neighbors(U, s, dist, k=1):
    """
    Find nearest neighbors. 
    Implement using prioprity queue.

    Parameters
    ----------
    U: array
        List of users.
    s: int
        Source user.
    dist: array
        Pairwise-distance.
    k: int, default = 1
        Number of neighbors.
    
    Returns
    -------
    T: array
        List of k nearest neighbors.
    """
    Q = []

    for i in U:
        heappush(Q, (dist[s][i], i))

    return [heappop(Q)[1] for i in range(k)]

def iterative_nearest_neighbor(U, S, dist):
    """
    Divide all users into equal-sized groups. 
    Implement using iterative nearest neighbor approach.

    Parameters
    ----------
    U: array
        List of users.
    S: int
        Group size.
    dist: array
        Pairwise-distance.

    Returns
    -------
    groups: dict
        Dictionary of groups.
    n_cls: int
        Number of groups.
    """
    length  = len(U)
    n_cls   = np.ceil(length/S)
    points  = set(U)
    groups  = dict()
    label   = 0

    while label < n_cls:
        avg_dist        = np.mean(dist[:, list(points)], axis=1)
        center          = np.argmax(avg_dist[list(points)])
        groups[label]   = find_nearest_neighbors(points, center, dist, k=min(S, len(points)))
        points.difference_update(groups[label])
        label        += 1
    
    return groups, n_cls

def find_nearest_POI(V, P, dist):
    """
    Find the nearest POI.

    Parameters
    ----------
    V: array
        List of nodes.
    P: array
        List of POIs.
    dist: array
        Pairwise-distance.
    
    Returns
    -------
    N: dict
        Dictionary of nearest POI .
    """
    N = dict()

    for v in V:
        min_dist    = np.inf
        tmp         = None
        for p in P:
            if dist[v][p] < min_dist:
                tmp         = p
                min_dist    = dist[v][p]
        N[v]        = tmp
    
    return N

def powerset(A):
    """
    Generate all subsets except the empty set.

    Parameters
    ----------
    A: array
        Universal set.
    
    Returns
    -------
    subsets: array
        List of subsets.
    """
    return list(chain.from_iterable(combinations(A, r) for r in range(1, len(A))))

def find_mp(M, comb, cost, travelDist, P, G, N, dist, epsilon):
    """
    Find the meeting points.
    
    Parameters
    ----------
    M: array
        List of confluent points.
    comb: array
        List of users.
    cost: dict
        Travel costs.
    travelDist: dict
        Travel distance of each user to the confluent points.
    B: list
        List of POIs.
    G: networkx graph
        Road network.
    N: dict
        Dictionary of nearest POI,
    dist: array
        Pairwise distance.
    epsilon: float
        Extra ratio.

    Returns
    -------
    X: array
        List of meeting points.
    cost: dict
        Travel costs.
    travelDist: dict
        Travel distance of each user to the meeting points.
    J: dict
        Dictionary of meeting points.
    p: int
        Nearest POI.
    minCost: int
        Travel cost to the nearest POI.
    """
    Q           = [] # node set
    J           = {} # meeting points dict
    X           = set() # NIMPs
    visited     = defaultdict(bool)

    # intialization
    for m in M:
        heappush(Q, (cost[m], m))
        J[(m, cost[m])] = m
        visited[m]      = True
    
    prev_cost   = -1
    prev_nodes  = set()
    p           = None
    minCost     = None
    
    while len(Q) > 0:
        # find the next node
        curr_cost, u = heappop(Q)
        if curr_cost > prev_cost:
            if prev_cost != -1: # check first node
                X = X.union(prev_nodes)
            prev_nodes  = {u}
        else:
            prev_nodes.add(u)
        prev_cost       = curr_cost
        if u in P: # terminate if a boundary node is found
            X.add(u)
            p           = u
            minCost     = curr_cost
            break

        for v in G.neighbors(u):
            if v not in X.union(prev_nodes):
                if not visited[v]:
                    cost[v]     = float('inf')
                    visited[v]  = True
                temp            = curr_cost + G.edges[u, v]['length']
                if cost[v] > temp:
                    # check detour constraint
                    violation       = False
                    for c in comb:
                        travelDist[(c, v, temp)]  = travelDist[(c, u, curr_cost)] + G.edges[u, v]['length']
                        if (travelDist[(c, v, temp)] + dist[v][N[v]]) > (1 + epsilon)*dist[c][N[c]]:
                            violation       = True
                            break
                    if not violation:
                        cost[v]     = temp
                        J[(v, cost[v])]     = J[(u, curr_cost)]
                        heappush(Q, (cost[v], v))

    return X, cost, travelDist, J, p, minCost

def compute_cost(U, P, G, z, N, dist, epsilon):
    """
    Compute the optimal travel plans for all possible combinations of less
    than or equal z users.

    Parameters
    ----------
    U: array
        List of users in a group.
    P: array
        List of POIs.
    G: networkx graph
        Road network.
    z: int
        Number of seats per car.
    N: dict
        Dictionary of nearest POI
    dist: array
        Pairwise distance.
    epsilon: float
        Extra ratio.

    Returns
    -------
    OptCost: dict
        Optimal travel cost.
    OptPOI: int
        Optimal POI.
    J: array
        Dictionary of meeting points.
    bestDiv: dict
        Best division of each combination to meeting points.
    bestSubgroup: dict
        Best subgroup of each combination to minimize travel cost.
    """
    V           = list(G.nodes())
    P           = set(P)
    X           = {} # NIMPs
    cost        = {} # travel cost
    J           = {} # meeting points
    OptPOI      = {} # optimal POIs
    OptCost     = {} # optimal cost
    travelDist  = {} # user costs

    # intialization
    for u in U:
        X[(u,)], cost[(u,)], travelDist[(u,)], J[(u,)], OptPOI[(u,)], OptCost[(u,)] = find_mp(
            [u], [u], {u: 0}, {(u, u, 0): 0}, P, G, N, dist, epsilon)

    # dynamic programming
    # loop through each combination of users
    M               = {} # meeting points
    bestSubgroup    = {} # subgrouping
    bestDiv         = {} # best division

    for j in range(2, z+1):
        for comb in combinations(U, j):
            X[comb]             = set()
            M[comb]             = set()
            OptCost[comb]       = float('inf')
            OptPOI[comb]        = None
            cost[comb]          = {}
            bestDiv[comb]       = {}
            travelDist[comb]    = {}
            for v in V:
                cost[comb][v]   = float('inf')
            all_combs           = powerset(comb)
            n_combs             = len(all_combs)
            for i in range(n_combs//2):
                comb1           = all_combs[i]
                comb2           = all_combs[n_combs-i-1]
                temp            = OptCost[comb1] + OptCost[comb2]
                if OptCost[comb] > temp:
                    OptCost[comb]       = temp
                    bestSubgroup[comb]  = (comb1, comb2)
                XX = X[comb1].intersection(X[comb2])
                for m in XX:
                    M[comb].add(m)
                    temp                = cost[comb1][m] + cost[comb2][m]
                    if cost[comb][m] > temp:
                        cost[comb][m]       = temp
                        bestDiv[comb][m]    = (comb1, comb2)
                        for c in comb1:
                            travelDist[comb][(c, m, temp)] = travelDist[comb1][(c, m, cost[comb1][m])]
                        for c in comb2:
                            travelDist[comb][(c, m, temp)] = travelDist[comb2][(c, m, cost[comb2][m])]    
            if len(M[comb]) > 0:
                X[comb], cost[comb], travelDist[comb], J[comb], p, temp = find_mp(
                    M[comb], comb, cost[comb], travelDist[comb], P, G, N, dist, epsilon)
                if (temp != None) and (OptCost[comb] > temp):
                    OptCost[comb]       = temp
                    OptPOI[comb]        = p
                    bestSubgroup[comb]  = (comb, None)
    
    return cost, OptCost, OptPOI, J, bestDiv, bestSubgroup

def compute_div(U, OptCost, z):
    """
    Compute the optimal division of a group into subgroups.

    Parameters
    ----------
    U: array
        List of users in a group.
    OptCost: dict
        Optimal travel cost.
    z: int
        Number of seats per car.

    Returns
    -------
    OptCost: dict
        Optimal travel cost.
    bestSubgroup: dict
        Best subgroup of each combination to minimize travel cost.
    """
    bestSubgroup    = {}

    for j in range(z+1, len(U)+1):
        for comb in combinations(U, j):
            OptCost[comb]   = float('inf')
            all_combs       = powerset(comb)
            n_combs         = len(all_combs)
            for i in range(n_combs//2):
                comb1           = all_combs[i]
                comb2           = all_combs[n_combs-i-1]
                temp            = OptCost[comb1] + OptCost[comb2]
                if OptCost[comb] > temp:
                    OptCost[comb]           = temp
                    bestSubgroup[comb]      = (comb1, comb2)
    
    return OptCost, bestSubgroup

def greedy_approach(U, N, dist):
    """
    Compute travel cost in the greedy approach, i.e.,
    each user travels to the nearest POI.

    Parameters
    ----------
    U: array
        List of users.
    N: dict
        Dictionary of nearest POI.
    dist: array
        Pairwise distance.
    
    Returns
    -------
    total_cost: int
        Total travel cost.
    """
    total_dist = 0

    for u in U:
        total_dist += dist[u][N[u]]
    
    return total_dist

def group_by_nearest_POI(U, N):
    """
    Divide all users into groups w.r.t their nearest POIs. 
    
    Parameters
    ----------
    U: array
        List of users.
    N: dict
        Dictionary of nearest POI.
    
    Returns
    -------
    groups: dict
        Dictionary of groups.
    """
    groups = defaultdict(list)

    for u in U:
        groups[N[u]].append(u) 
    
    return groups

def cal_travel_cost(group, bestSubgroup, OptCost):
    """
    Compute total travel cost of the group 
    in accordance to the optimal travel plan.

    Parameters
    ----------
    group: array
        List of users.
    bestSubgroup: dict
        Best subgroup of each combination to minimize travel cost.
    OptCost: dict
        Optimal travel cost.
    
    Returns
    -------
    cost: int
        Total travel cost.
    """
    cost    = 0

    if tuple(group) in bestSubgroup:
        for comb in bestSubgroup[tuple(group)]:
            cost += OptCost[comb]
    else: # the case when number of users in the last group <= z.
        cost = OptCost[tuple(group)]

    return cost