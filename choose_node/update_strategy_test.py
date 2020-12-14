
import numpy as np
import geatpy as ea
from hsdn_near_optimal_performance import SOHybridNetTEOptimizeProblem
import project_xml_reader
import os
from concurrent.futures import ProcessPoolExecutor
import json

max_val = float('inf')
max_utilization_dict = {}
utilization_func_val_dict = {}
variance_dict = {}
congestion_times_dict = {}
base_dir = "../abilene/TM/2004"
num_to_city_dict = {2: "ATLA-M5", 3: "ATLAng", 11: "CHINng", 8: "DNVRng", 4: "HSTNng", 10: "IPLSng",
                    9: "KSCYng", 5: "LOSAng", 0: "NYCMng", 6: "SNVAng", 7: "STTLng", 1: "WASHng"}
city_to_num_dict = {"ATLA-M5": 2, "ATLAng": 3, "CHINng": 11, "DNVRng": 8, "HSTNng": 4, "IPLSng": 10,
                    "KSCYng": 9, "LOSAng": 5, "NYCMng": 0, "SNVAng": 6, "STTLng": 7, "WASHng": 1}


def calc_normal_utilization(graph, sdn_count, sdn_nodes, bandwidth):
    exclude_dir = []
    for month_dir in os.listdir(base_dir):
        if month_dir in exclude_dir:
            continue
        month_dir_abs_path = os.path.join(base_dir, month_dir)
        solve_segments(month_dir_abs_path, graph, sdn_count, sdn_nodes, bandwidth, 2)
    print(congestion_times_dict)


def solve_segments(dir_name, graph, sdn_count, sdn_nodes, bandwidth, worker_count):
    print(dir_name)
    global max_utilization_dict
    global congestion_times_dict
    global utilization_func_val_dict
    global variance_dict
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        jobs = {}
        for traffic_xml in os.listdir(dir_name):
            jobs[traffic_xml] = executor.submit(solve_one_file, graph, sdn_count, sdn_nodes, bandwidth,
                                                os.path.join(dir_name, traffic_xml))
        for job_key in jobs.keys():
            utilization, max_x, max_y, max_utilization_formula_val, min_variance = jobs[job_key].result()
            max_utilization_dict[job_key] = utilization
            utilization_func_val_dict[job_key] = max_utilization_formula_val
            variance_dict[job_key] = min_variance
            cong_key = num_to_city_dict[max_x] + "->" + num_to_city_dict[max_y]
            if utilization > 0.9:
                if cong_key in congestion_times_dict.keys():
                    congestion_times_dict[cong_key] = congestion_times_dict[cong_key] + 1
                else:
                    congestion_times_dict[cong_key] = 1
    out_dir = "upgrade_strategy_%s_node%s" % (sdn_count, "" if sdn_count <= 1 else "s")
    utilization_json_name = ''.join(['utilization/%s/' % out_dir,
                                     dir_name.replace('\\', '_').replace('/', '_'), '.json'])
    utilization_func_val_json_name = ''.join(['utilization_function_value/%s/' % out_dir,
                                              dir_name.replace('\\', '_').replace('/', '_'), '.json'])
    variance_json_name = ''.join(['variance/%s/' % out_dir,
                                  dir_name.replace('\\', '_').replace('/', '_'), '.json'])
    f = open(utilization_json_name, mode='w', encoding='utf-8')
    f.write(json.dumps(max_utilization_dict))
    f.close()
    max_utilization_dict.clear()
    f = open(utilization_func_val_json_name, mode='w', encoding='utf-8')
    f.write(json.dumps(utilization_func_val_dict))
    f.close()
    utilization_func_val_dict.clear()
    f = open(variance_json_name, mode='w', encoding='utf-8')
    f.write(json.dumps(variance_dict))
    f.close()
    variance_dict.clear()


def solve_one_file(graph, sdn_count, sdn_nodes, bandwidth, filename):
    traffic = project_xml_reader.parse_traffics(filename)
    # 区域描述
    problem = SOHybridNetTEOptimizeProblem(graph, sdn_count, traffic, bandwidth, filename)
    Encodings = ['P', 'RI']
    Field1 = ea.crtfld(Encodings[0], problem.varTypes[:sdn_count],
                       problem.ranges[:, :sdn_count], problem.borders[:, :sdn_count])  # 创建区域描述器
    Field2 = ea.crtfld(Encodings[1], problem.varTypes[sdn_count:],
                       problem.ranges[:, sdn_count:], problem.borders[:, sdn_count:])  # 创建区域描述器
    Fields = [Field1, Field2]
    weight_size = int(np.sum(graph == 1) / 2)
    # weights = [233, 846, 1, 1176, 1893, 366, 861, 1295, 2095, 902, 639, 587, 548, 700, 260]
    weights = [1] * weight_size
    pop = ea.PsyPopulation(Encodings, Fields, 1, Phen=np.array([sdn_nodes + weights]))

    return problem.aimFunc1(pop)


def optimize_link_utilization(graph, sdn_count, sdn_nodes, bandwidth, month_index, json_name, threshold=0.5):
    f = open(json_name, 'r', encoding='utf-8')
    utilization_dict = dict(json.load(f))
    f.close()
    for file_name in utilization_dict.keys():
        if utilization_dict[file_name] >= threshold:
            print(file_name, ": ", utilization_dict[file_name])
            file_abs_path = os.path.join(base_dir, month_index, file_name)
            solve_one_file(graph, sdn_count, sdn_nodes, bandwidth, file_abs_path)


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
    sdn_count = 2
    # [11, 10, 9, 6, 3, 8, 1, 7, 5, 4, 0, 2]
    # sdn_node_permutation:  [10 11  9  3  6  2  8  1  5  4  0  7]
    sdn_nodes = [10, 11]
    calc_normal_utilization(graph, sdn_count, sdn_nodes, bandwidth)  # 计算所有流量的链路利用率
    # "TM-2004-06-02-1815.xml": 1.117167215658603,
    # solve_one_file(graph, sdn_count, sdn_nodes, bandwidth, "abilene/TM/2004/09/TM-2004-09-01-0620.xml")
    # optimize_link_utilization(graph, sdn_count, sdn_nodes, bandwidth, '04', 'utilization/abilene_TM_2004_04.json')
