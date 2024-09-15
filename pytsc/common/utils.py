import logging
import re

import numpy as np


class EnvLogger:
    logger = None

    @staticmethod
    def _ensure_handler(logger):
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    @staticmethod
    def get_logger(name=__name__, level=logging.INFO):
        if EnvLogger.logger is None:
            EnvLogger.logger = logging.getLogger(name)
            EnvLogger._ensure_handler(EnvLogger.logger)
            EnvLogger.logger.setLevel(level)  # Set the logger level dynamically
        return EnvLogger.logger

    @staticmethod
    def log(level, msg):
        logger = EnvLogger.get_logger()
        logger.log(level, msg)

    @staticmethod
    def log_info(msg):
        EnvLogger.log(logging.INFO, msg)

    @staticmethod
    def log_warning(msg):
        EnvLogger.log(logging.WARNING, msg)

    @staticmethod
    def set_log_level(level):
        logger = EnvLogger.get_logger()
        logger.setLevel(level)


def validate_input_against_allowed(inp_cfg_str, allowed_cfg_strs):
    assert (
        inp_cfg_str in allowed_cfg_strs
    ), f"Invalid action space `{inp_cfg_str}`. Allowed values are {allowed_cfg_strs}."


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


def get_vehicle_bin_index(n_bins, lane_length, vehicle_position):
    if vehicle_position < 0:
        vehicle_position = 0
    elif vehicle_position > lane_length:
        vehicle_position = lane_length
    bin_size = lane_length / n_bins
    bin_index = int(vehicle_position // bin_size)
    if bin_index >= n_bins:
        bin_index = n_bins - 1
    return bin_index


def calculate_bin_index(n_bins, bin_size, lane_length, lane_position):
    visibility_length = n_bins * bin_size
    distance_from_intersection = lane_length - lane_position
    if distance_from_intersection > visibility_length:
        return None  # The vehicle is not within the visible section
    # Calculate the bin index
    # NOTE: Bins are indexed from the intersection backwards
    bin_index = (visibility_length - distance_from_intersection) // bin_size
    # Ensure the bin index is within the expected range [0, 9] for 10 bins
    bin_index = min(max(bin_index, 0), n_bins - 1)
    return int(bin_index)


# def map_position_to_matrix(position, x_min, y_min, resolution):
#     """
#     Maps a position (x, y) to the corresponding indices in the matrix
#     """
#     x, y = position
#     row = int((y - y_min) // resolution)
#     col = int((x - x_min) // resolution)
#     return row, col


def compute_linearly_weighted_average(position_matrices):
    """
    position_matrices: dequeue
    """
    n = len(position_matrices)
    lwma = np.zeros_like(position_matrices[0])
    normalization_factor = n * (n + 1) / 2
    for i, matrix in enumerate(reversed(position_matrices)):
        weight = (n - i) / normalization_factor
        lwma += weight * matrix
    return np.round(lwma, 3)


def generate_weibull_flow_rates(shape, scale, max_rate, num_segments):
    inter_arrival_times = np.random.weibull(shape, 1000) * scale
    cumulative_times = np.cumsum(inter_arrival_times)
    cumulative_times = cumulative_times[cumulative_times <= 3600]
    random_peak_segment = np.random.randint(0, num_segments)
    x = np.linspace(-2, 2, num_segments)
    flow_rates = np.exp(-(x**2))  # Gaussian-like curve
    flow_rates = flow_rates / max(flow_rates) * max_rate
    flow_rates = np.roll(flow_rates, random_peak_segment)
    return flow_rates
