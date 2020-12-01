import numpy as np
import json
import project_xml_reader
import os
import graph_util as gu
import near_optimal_split_ratio as nosr
import calculator

max_val = float('inf')


def get_traffics(json_name, traffic_dir):
    json_file = open(json_name)
    link_max_utilization_json = json.load(json_file, object_hook=dict)
    traffics = []
    for file in link_max_utilization_json.keys():
        if link_max_utilization_json[file] >= 0.4:
            traffics.append(project_xml_reader.parse_traffics(os.path.join(traffic_dir, file)))
    return traffics


if __name__ == "__main__":
    """
    固定权重和节点
    """
    graph = np.ones([12, 12]) * max_val
    graph[range(len(graph)), range(len(graph))] = 0
    graph[0][1], graph[0][11], graph[1][0], graph[1][3], graph[2][3], graph[3][1], graph[3][2], graph[3][4], \
        graph[3][10], graph[4][3], graph[4][5], graph[4][9], graph[5][4], graph[5][6], graph[6][5], \
        graph[6][7], graph[6][8], graph[7][6], graph[7][8], graph[8][7], graph[8][6], graph[8][9], \
        graph[9][8], graph[9][4], graph[9][10], graph[10][9], graph[10][3], graph[10][11], graph[11][10], graph[11][0] \
        = [1] * 30
    bandwidth = np.zeros([12, 12])
    bandwidth[0][1], bandwidth[0][11], bandwidth[1][3], bandwidth[2][3], bandwidth[3][4], bandwidth[4][5], \
        bandwidth[4][9], bandwidth[5][6], bandwidth[6][7], bandwidth[6][8], bandwidth[7][8], bandwidth[8][9], \
        bandwidth[9][10], bandwidth[10][11] = [9920000] * 14
    bandwidth[3][10] = 2480000
    json_name = "../utilization/add_weight/abilene_TM_2004_05.json"
    traffic_dir = "../abilene/TM/2004/05/"
    traffics = get_traffics(json_name, traffic_dir)
    sdn_nodes = [10, 11]
    filled_weight_list = graph.copy()
    shortest_path_list = [gu.dijkstra_alg(filled_weight_list, i) for i in range(12)]
    total_bandwidth_used = np.zeros([12, 12])
    for index in range(12):
        # 先将每个节点看成传统节点，以每个顶点为目标节点，构建有向无环图
        legacy_node_dag = gu.build_dag(filled_weight_list, index, shortest_path_list)
        # 针对每一个顶点的有向无环图查找sdn节点，增加可用链路并验证环路
        dag, sorted_nodes = gu.add_links(filled_weight_list, legacy_node_dag, index, sdn_nodes)
        near_optimal_bandwidth_used = nosr.execute(dag, sorted_nodes, traffics, bandwidth, sdn_nodes)
        total_bandwidth_used += near_optimal_bandwidth_used
    utilization_formula_val = calculator.calc_utilization_formula(bandwidth,
                                                                  total_bandwidth_used, True)
    variance = calculator.calc_remaining_bandwidth_variance(bandwidth, total_bandwidth_used)
    max_utilization = calculator.calc_max_utilization(bandwidth, total_bandwidth_used)[0]
    print()
