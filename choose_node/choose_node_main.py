import geatpy as ea
import numpy as np
import project_xml_reader
from choose_node.determined_weight_optimize import NearOptLinkParamsWithDeterminedWeight

max_val = float('inf')


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
    sdn_count = 3
    traffic = project_xml_reader.parse_traffics("abilene/TM/2004/09/TM-2004-09-01-0620.xml")
    # 区域描述
    problem = NearOptLinkParamsWithDeterminedWeight(graph, sdn_count, traffic, bandwidth, "TM-2004-09-01-0620.xml")
    Encoding = 'P'
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    sdn_nodes = [3, 4, 11]
    NIND = 50
    myAlgorithm = ea.moea_NSGA3_DE_templet(problem, Encoding)
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm.mutOper.F = 0.74
    myAlgorithm.recOper.XOVR = 0.65
    myAlgorithm.MAXGEN = 40
    myAlgorithm.drawing = 2
    myAlgorithm.run()
    pop = ea.Population(Encoding, Field, 1, Phen=np.array([sdn_nodes]))
