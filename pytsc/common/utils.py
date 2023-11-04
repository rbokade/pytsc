import itertools
import re

import numpy as np


def sort_alphanumeric_ids(ids):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(ids, key=alphanum_key)


def recursively_update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursively_update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def flatten_list(list_of_lists):
    flat_list = []
    # Iterate through the outer list
    for element in list_of_lists:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def pad_array(array, size, pad_value=0):
    """
    Pads the given array with the specified padding value so that it has
    the specified size. If the array is already larger than the specified size,
    it is returned unmodified.
    """
    if len(array) >= size:
        return array
    else:
        padded_arr = np.full(size, pad_value)
        padded_arr[: len(array)] = array
        return padded_arr


def pad_list(inp_list, size, pad_value=0):
    return pad_array(np.array(inp_list), size, pad_value).tolist()


def map_to_phase_circle(phase_index, time_on_phase, phase_durations):
    """
    Maps the time on a given phase to its corresponding position on a
    phase circle.
    """
    # Compute the starting angle of each phase on the phase circle
    total_duration = sum(phase_durations)
    start_angles = [
        2 * np.pi * sum(phase_durations[:i]) / total_duration
        for i in range(len(phase_durations))
    ]
    # Compute the angular span of the current phase
    phase_span = 2 * np.pi * phase_durations[phase_index] / total_duration
    # Compute the angular position within the current phase
    phase_position = phase_span * (
        time_on_phase / phase_durations[phase_index]
    )
    return start_angles[phase_index] + phase_position


def compute_local_order_for_agent(agent_phase_angles, adj_row):
    cos_values = np.cos(agent_phase_angles)
    sin_values = np.sin(agent_phase_angles)
    mean_cos = np.sum(cos_values * adj_row) / np.sum(adj_row)
    mean_sin = np.sum(sin_values * adj_row) / np.sum(adj_row)
    return np.sqrt(mean_cos**2 + mean_sin**2)
