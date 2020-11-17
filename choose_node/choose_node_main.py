import geatpy as ea
import numpy as np
import project_xml_reader
from choose_node.determined_weight_optimize import NearOptUpgradeStrategyWithDeterminedWeight
import json
import os

max_val = float('inf')
f = open("5月优化9月数据.txt", 'w', encoding="utf-8")


def do_verify_result(problem, perm_list):
    result_weight = [1, 0]
    optimal_solution = []
    global_min = max_val
    for sdn_nodes in perm_list:
        performance = problem.solve_one_pop(sdn_nodes, True)
        curr_min = performance[0] * result_weight[0] + performance[1] * result_weight[1]
        if curr_min < global_min:
            optimal_solution = sdn_nodes
    print(optimal_solution, file=f)


def get_traffics(threshold=0.7):
    json_file = open("../ecmp_utilization/add_weight/abilene_TM_2004_09.json")
    base_dir = "../abilene/TM/2004/09/"
    link_max_utilization_json = json.load(json_file, object_hook=dict)
    traffics = []
    origin_mean_utilization = 0
    occurred_times = 0
    for file in link_max_utilization_json.keys():
        if link_max_utilization_json[file] >= threshold:
            origin_mean_utilization += link_max_utilization_json[file]
            occurred_times += 1
            traffics.append(project_xml_reader.parse_traffics(os.path.join(base_dir, file)))
    origin_mean_utilization /= occurred_times
    print("origin mean utilization: ", origin_mean_utilization, file=f)
    return traffics


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
        graph[9][8], graph[9][4], graph[9][10], graph[10][9], graph[10][3], graph[10][11], graph[11][10], graph[11][0] \
        = [1] * 30
    bandwidth = np.zeros([12, 12])
    bandwidth[0][1], bandwidth[0][11], bandwidth[1][3], bandwidth[2][3], bandwidth[3][4], bandwidth[4][5], \
        bandwidth[4][9], bandwidth[5][6], bandwidth[6][7], bandwidth[6][8], bandwidth[7][8], bandwidth[8][9], \
        bandwidth[9][10], bandwidth[10][11] = [9920000] * 14
    bandwidth[3][10] = 2480000
    # traffic = project_xml_reader.parse_traffics("../abilene/TM/2004/09/TM-2004-09-01-0620.xml")
    # 区域描述
    problem = NearOptUpgradeStrategyWithDeterminedWeight(graph, get_traffics(), bandwidth)
    Encoding = 'P'
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    NIND = 50
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm = ea.moea_NSGA3_templet(problem, population)
    myAlgorithm.XOVR = 0.65
    myAlgorithm.Pm = 0.7
    myAlgorithm.selFunc = 'tour'
    myAlgorithm.MAXGEN = 30
    myAlgorithm.drawing = 2
    # NDSet = myAlgorithm.run()
    perm_list = np.array([[11, 10, 9, 6, 3, 8, 1, 7, 5, 4, 0, 2]])
    # print(perm_list)
    do_verify_result(problem, perm_list)
    # NDSet.save()
    # sdn_nodes = [3]
    # pop = ea.Population(Encoding, Field, 1, Phen=np.array([sdn_nodes]))
    # problem.aimFunc(pop)
