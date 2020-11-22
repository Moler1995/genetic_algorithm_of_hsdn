import geatpy as ea
import numpy as np
import project_xml_reader
from choose_node.determined_weight_optimize import NearOptUpgradeStrategyWithDeterminedWeight
import json
import os

max_val = float('inf')


def do_verify_result(problem, Phen):
    result_weight = [1, 0]
    sdn_nodes_list = Phen
    optimal_solution = []
    global_min = max_val
    for sdn_nodes in sdn_nodes_list:
        performance = problem.solve_one_pop(sdn_nodes, True)
        curr_min = performance[0] * result_weight[0] + performance[1] * result_weight[1]
        if curr_min < global_min:
            optimal_solution = sdn_nodes
    print(optimal_solution)


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
    json_name = "../utilization/add_weight/abilene_TM_2004_05.json"
    traffic_dir = "../abilene/TM/2004/05/"
    # 5月份的数据跑的结果[11,10,9,6,3,8,1,7,5,4,0,2]
    # [11 10  9  3  4  5  0  6  1  7  8  2] result:  (129170827.83677036, 2103619.324897012, 0.49827735384980926)
    problem = NearOptUpgradeStrategyWithDeterminedWeight(graph, get_traffics(json_name, traffic_dir), bandwidth)
    Encoding = 'P'
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    NIND = 50
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm = ea.moea_NSGA3_templet(problem, population)
    myAlgorithm.XOVR = 0.65
    myAlgorithm.Pm = 0.7
    myAlgorithm.selFunc = 'tour'
    myAlgorithm.MAXGEN = 50
    myAlgorithm.drawing = 1
    NDSet = myAlgorithm.run()
    # solution_to_verify = np.array([[11, 10, 9, 6, 3, 8, 1, 7, 5, 4, 0, 2]])
    # do_verify_result(problem, solution_to_verify).
    NDSet.save()
    # sdn_nodes = [3]
    # pop = ea.Population(Encoding, Field, 1, Phen=np.array([sdn_nodes]))
    # problem.aimFunc(pop)
