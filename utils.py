import os
from datetime import datetime
import numpy as np

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
        np.save(f, np.array(run_time))
    with open(cost_path, 'wb') as f:
        np.save(f, np.array(travel_cost))