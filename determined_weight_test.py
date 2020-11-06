
import numpy as np
import geatpy as ea
from hsdn_near_optimal_performance import SOHybridNetTEOptimizeProblem
import project_xml_reader
import os

max_val = float('inf')
max_utilization_dict = {}

if __name__ == "__main__":
    """
    固定权重，逐次增加部署比例
    节点升级选择策略为直连链路最多的几个节点,如果直连链路数量相同则对比所有直连链路的带宽和
    """
    graph = np.ones([12, 12]) * max_val
    graph[range(len(graph)), range(len(graph))] = 0
    graph[0][1], graph[0][11], graph[1][0], graph[1][3], graph[2][3], graph[3][1], graph[3][2], graph[3][4], \
        graph[3][10], graph[4][3], graph[4][5], graph[4][9], graph[5][4], graph[5][6], graph[6][5], \
        graph[6][7], graph[6][8], graph[7][6], graph[7][8], graph[8][7], graph[8][6], graph[8][9], \
        graph[9][8], graph[9][4], graph[9][10], graph[10][9], graph[10][3], graph[10][11], graph[11][10], graph[11][0]\
        = [1] * 30
    bandwidth = np.zeros([12, 12])
    bandwidth[0][1], bandwidth[0][11], bandwidth[1][3], bandwidth[2][3], bandwidth[3][4], bandwidth[4][5], \
        bandwidth[4][9], bandwidth[5][6], bandwidth[6][7], bandwidth[6][8], bandwidth[7][8], bandwidth[8][9], \
        bandwidth[9][10], bandwidth[10][11] = [9920000] * 14
    bandwidth[3][10] = 2480000
    sdn_count = 0
    sdn_nodes = []
    max_direct_link = 0
    for i in range(sdn_count):
        sdn_node = 0
        for j in range(len(graph)):
            if j in sdn_nodes:
                continue
            direct_link = np.sum(graph[j] == 1)
            if direct_link > max_direct_link:
                max_direct_link = direct_link
                sdn_node = j
            elif direct_link == max_direct_link:
                if np.sum(bandwidth[j]) > np.sum(bandwidth[sdn_node]):
                    sdn_node = j
        sdn_nodes.append(sdn_node)
        max_direct_link = 0
    # sdn_nodes = [4, 6, 10]
    for month_dir in os.listdir("abilene/TM/2004"):
        month_dir_abs_path = os.path.join("abilene/TM/2004", month_dir)
        for traffic_xml in os.listdir(month_dir_abs_path):
            traffic = project_xml_reader.parse_traffics(os.path.join(month_dir_abs_path, traffic_xml))
            # 区域描述
            problem = SOHybridNetTEOptimizeProblem(graph, sdn_count, traffic, bandwidth, traffic_filename=traffic_xml)
            Encodings = ['P', 'RI']
            Field1 = ea.crtfld(Encodings[0], problem.varTypes[:sdn_count],
                               problem.ranges[:, :sdn_count], problem.borders[:, :sdn_count])  # 创建区域描述器
            Field2 = ea.crtfld(Encodings[1], problem.varTypes[sdn_count:],
                               problem.ranges[:, sdn_count:], problem.borders[:, sdn_count:])  # 创建区域描述器
            Fields = [Field1, Field2]
            weight_size = int(np.sum(graph == 1) / 2)
            weights = [1] * weight_size
            pop = ea.PsyPopulation(Encodings, Fields, 1, Phen=np.array([sdn_nodes + weights]))
            problem.aimFunc(pop)

    print(max_utilization_dict)

