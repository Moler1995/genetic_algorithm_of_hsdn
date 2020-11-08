
import numpy as np
import geatpy as ea
from hsdn_near_optimal_performance import SOHybridNetTEOptimizeProblem
import project_xml_reader
import os
from concurrent.futures import ProcessPoolExecutor
import json

max_val = float('inf')
max_utilization_dict = {}


def calc_normal_utilization(graph, sdn_count, sdn_nodes, bandwidth):
    exclude_dir = ['03', '04', '05']
    for month_dir in os.listdir("abilene/TM/2004"):
        if month_dir in exclude_dir:
            continue
        month_dir_abs_path = os.path.join("abilene/TM/2004", month_dir)
        solve_segments(month_dir_abs_path, graph, sdn_count, sdn_nodes, bandwidth, 10)


def solve_segments(dir_name, graph, sdn_count, sdn_nodes, bandwidth, worker_count):
    print(dir_name)
    global max_utilization_dict
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        jobs = {}
        for traffic_xml in os.listdir(dir_name):
            jobs[traffic_xml] = executor.submit(solve_one_file, graph, sdn_count, sdn_nodes, bandwidth,
                                                os.path.join(dir_name, traffic_xml))
        for job_key in jobs.keys():
            max_utilization_dict[job_key] = jobs[job_key].result()
    json_name = ''.join(['ecmp_utilization/', dir_name.replace('\\', '_').replace('/', '_'), '.json'])
    f = open(json_name, mode='w', encoding='utf-8')
    f.write(json.dumps(max_utilization_dict))
    f.close()


def solve_one_file(graph, sdn_count, sdn_nodes, bandwidth, filename):
    traffic = project_xml_reader.parse_traffics(filename)
    # 区域描述
    problem = SOHybridNetTEOptimizeProblem(graph, sdn_count, traffic, bandwidth)
    Encodings = ['P', 'RI']
    Field1 = ea.crtfld(Encodings[0], problem.varTypes[:sdn_count],
                       problem.ranges[:, :sdn_count], problem.borders[:, :sdn_count])  # 创建区域描述器
    Field2 = ea.crtfld(Encodings[1], problem.varTypes[sdn_count:],
                       problem.ranges[:, sdn_count:], problem.borders[:, sdn_count:])  # 创建区域描述器
    Fields = [Field1, Field2]
    weight_size = int(np.sum(graph == 1) / 2)
    weights = [1] * weight_size
    pop = ea.PsyPopulation(Encodings, Fields, 1, Phen=np.array([sdn_nodes + weights]))

    return problem.aimFunc(pop)


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
    sdn_count = 12
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
    sdn_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # calc_normal_utilization(graph, sdn_count, sdn_nodes, bandwidth) # 计算所有流量的链路利用率
    # "TM-2004-06-02-1815.xml": 1.117167215658603,
    solve_one_file(graph, sdn_count, sdn_nodes, bandwidth, "abilene/TM/2004/06/TM-2004-06-02-1815.xml")

