# Import libraries
import numpy as np
from heapq import heappush, heappop

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
    dist: dict
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

    Returns
    -------
    cls: dict
        Dictionary of groups.
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