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
