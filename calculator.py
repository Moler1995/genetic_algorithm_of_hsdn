import numpy as np
from fractions import Fraction

max_val = float('inf')


def calc_utilization_formula(total_band_width, used_bandwidth):
    filled_bandwidth = total_band_width.copy()
    filled_bandwidth[filled_bandwidth == 0.0] = max_val
    utilization_matrix = used_bandwidth / filled_bandwidth
    # print(utilization_matrix)
    max_utilization = np.max(utilization_matrix)
    max_x_index, max_y_index = np.unravel_index(np.argmax(utilization_matrix), utilization_matrix.shape)
    max_utilization_bandwidth_used = used_bandwidth[max_x_index][max_y_index]
    max_utilization_raw_bandwidth = total_band_width[max_x_index][max_y_index]
    if 0 <= max_utilization <= Fraction(1, 3):
        return max_utilization_bandwidth_used
    elif Fraction(1, 3) < max_utilization <= Fraction(2, 3):
        return 3 * max_utilization_bandwidth_used - Fraction(2, 3) * max_utilization_raw_bandwidth
    elif Fraction(2, 3) < max_utilization <= Fraction(9, 10):
        return 10 * max_utilization_bandwidth_used - Fraction(16, 3) * max_utilization_raw_bandwidth
    elif Fraction(9, 10) < max_utilization <= 1:
        return 70 * max_utilization_bandwidth_used - Fraction(178, 3) * max_utilization_raw_bandwidth
    elif 1 < max_utilization <= Fraction(11, 10):
        return 500 * max_utilization_bandwidth_used - Fraction(1468, 3) * max_utilization_raw_bandwidth
    else:
        return 5000 * max_utilization_bandwidth_used - Fraction(16318, 3) * max_utilization_raw_bandwidth


def calc_remaining_bandwidth_variance(total_band_width, used_bandwidth):
    remaining_bandwidth = total_band_width - used_bandwidth
    direct_link_count = np.count_nonzero(total_band_width)
    avg_remaining_bandwidth = np.sum(remaining_bandwidth) / direct_link_count
    node_count = len(total_band_width)
    variance_sum = 0
    for i in range(node_count):
        for j in range(node_count):
            # 判断一下两点之间是否有直接链接
            if total_band_width[i][j] != 0:
                variance_sum += (remaining_bandwidth[i][j] - avg_remaining_bandwidth) ** 2
    return variance_sum / direct_link_count
