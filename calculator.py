import numpy as np
from fractions import Fraction

max_val = float('inf')


def calc_utilization_formula(total_band_width, used_bandwidth, do_print=False):
    """
    计算总的链路利用率标准函数
    :param total_band_width: 初始带宽
    :param used_bandwidth:  使用带宽
    :param do_print: 是否打印带宽利用率
    :return:
    """
    filled_bandwidth = total_band_width.copy()
    filled_bandwidth[filled_bandwidth == 0.0] = max_val
    utilization_matrix = used_bandwidth / filled_bandwidth
    if do_print:
        print(utilization_matrix)
        # print(np.max(utilization_matrix))
    # max_utilization = np.max(utilization_matrix)
    max_x_index, max_y_index = np.unravel_index(np.argmax(utilization_matrix), utilization_matrix.shape)
    # max_utilization_bandwidth_used = used_bandwidth[max_x_index][max_y_index]
    # max_utilization_raw_bandwidth = total_band_width[max_x_index][max_y_index]
    target_val = 0
    for i in range(len(total_band_width)):
        for j in range(len(total_band_width)):
            band_width_used = used_bandwidth[i][j]
            utilization = utilization_matrix[i][j]
            raw_bandwidth = total_band_width[i][j]
            if 0 <= utilization <= Fraction(1, 3):
                target_val += band_width_used
            elif Fraction(1, 3) < utilization <= Fraction(2, 3):
                target_val += 3 * band_width_used - Fraction(2, 3) * raw_bandwidth
            elif Fraction(2, 3) < utilization <= Fraction(9, 10):
                target_val += 10 * band_width_used - Fraction(16, 3) * raw_bandwidth
            elif Fraction(9, 10) < utilization <= 1:
                target_val += 70 * band_width_used - Fraction(178, 3) * raw_bandwidth
            elif 1 < utilization <= Fraction(11, 10):
                target_val += 500 * band_width_used - Fraction(1468, 3) * raw_bandwidth
            else:
                target_val += 5000 * band_width_used - Fraction(16318, 3) * raw_bandwidth
    return target_val


def calc_max_utilization(total_band_width, used_bandwidth):
    """
    计算总的链路利用率标准函数
    :param total_band_width: 初始带宽
    :param used_bandwidth:  使用带宽
    :return:
    """
    filled_bandwidth = total_band_width.copy()
    filled_bandwidth[filled_bandwidth == 0.0] = max_val
    utilization_matrix = used_bandwidth / filled_bandwidth
    max_utilization = np.max(utilization_matrix)
    max_x_index, max_y_index = np.unravel_index(np.argmax(utilization_matrix), utilization_matrix.shape)
    return max_utilization, max_x_index, max_y_index


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
    return (variance_sum / direct_link_count) ** (1 / 2)
