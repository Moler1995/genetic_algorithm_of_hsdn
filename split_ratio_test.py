import random
from collections import deque

import geatpy as ea
import numpy as np

from near_optimal_split_ratio import NearOptimalSplitRatioProblem

max_val = float('inf')

"""
测试
"""
if __name__ == "__main__":
    dag = np.array([[0, 2, max_val, max_val], [max_val, 0, max_val, 3], [1, 1, 0, 4], [max_val, max_val, max_val, 0]])
    topological_sorted_nodes = deque([2, 0, 1, 3])
    traffic = np.array([[0.0, 0.0, 0.0, 9.0], [0.0, 0.0, 0.0, 8.0], [0.0, 0.0, 0.0, 12.0], [0.0, 0.0, 0.0, 0.0]])
    bandwidth = np.array([[0.0, 15.0, 20.0, 0.0], [0.0, 0.0, 20.0, 30.0], [0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
    sdn_nodes = [1, 2]
    problem = NearOptimalSplitRatioProblem(dag=dag, topological_sorted_nodes=topological_sorted_nodes,
                                           sdn_nodes=sdn_nodes, band_width=bandwidth, traffics=np.array([traffic]))
    Encoding = "RI"
    NIND = 80
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    # myAlgorithm = ea.moea_NSGA2_templet(problem, population) # 没算出结果，再看看
    myAlgorithm = ea.moea_NSGA3_DE_templet(problem, population) # 有结果，再分析看看
    # myAlgorithm = ea.moea_NSGA2_DE_templet(problem, population) # 有结果，再对比分析看看
    # myAlgorithm = ea.moea_MOEAD_DE_templet(problem, population) # 结果比较诡异，需要分析，执行时间长
    # myAlgorithm.mutOper.F = 0.72  # 4/20
    # myAlgorithm.mutOper.F = 0.73 # 1/20
    myAlgorithm.mutOper.F = 0.74  # 1/40
    # myAlgorithm.mutOper.F = 0.75  # 2/20
    # myAlgorithm.mutOper.F = 0.745  # 2/20
    # myAlgorithm.mutOper.F = 0.742 # 2/20
    # myAlgorithm.mutOper.F = 0.741  # 1/20
    # myAlgorithm.recOper.XOVR = 0.65
    # myAlgorithm.mutOper.Pm = 0.7  # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.65  # 修改交叉算子的交叉概率
    # myAlgorithm = ea.moea_awGA_templet(problem, population) # 结果比较诡异，需要分析
    # myAlgorithm = ea.moea_MOEAD_archive_templet(problem, population) # 没结果
    # myAlgorithm = ea.moea_MOEAD_templet(problem, population) # 没算出结果，再看看
    # myAlgorithm = ea.moea_NSGA3_templet(problem, population) # 没算出结果，再看看

    # 自定义初始种群,计算目标函数值和约束
    initChrom = []
    unit_ratio = 1 / NIND
    chrom = []
    for j in range(0, NIND):
        start_index = 0
        link_count = 0
        for sdn_node in sdn_nodes:
            if sdn_node not in problem.sdn_node_link_count.keys():
                continue
            link_count = problem.sdn_node_link_count[sdn_node]
            for index in range(link_count):
                if index == link_count - 1:
                    chrom.append(NIND - sum(chrom[start_index:start_index + index]))
                    initChrom.append(chrom.copy())
                    chrom.clear()
                    start_index += link_count
                else:
                    chrom.append(random.randint(0, NIND - sum(chrom[start_index:start_index + index])))
    prophetPop = ea.Population(Encoding, Field, NIND,
                               np.array(initChrom) * unit_ratio, Phen=np.array(initChrom) * unit_ratio)
    problem.aimFunc(prophetPop)
    # used = problem.route_flow([0.079, 0.315, 0.606])
    # print(calculator.calc_remaining_bandwidth_variance(bandwidth, used))
    # print(calculator.calc_utilization_formula(bandwidth, used))
    myAlgorithm.MAXGEN = 100
    myAlgorithm.drawing = 2
    NDSet = myAlgorithm.run(prophetPop)
    print('用时：%s 秒' % myAlgorithm.passTime)
    print('非支配个体数：%s 个' % NDSet.sizes)
    # target_val = NDSet.ObjV[:, 0] * 0.4 + NDSet.ObjV[:, 1] * 0.6
    # min_index = np.argmin(target_val)
    # print(NDSet.ObjV[min_index], min_index, NDSet.Phen[min_index])
    # for i in range(0, 20):
    #     myAlgorithm.MAXGEN = 100
    #     myAlgorithm.drawing = 2
    #     NDSet = myAlgorithm.run(prophetPop)
    #     print('用时：%s 秒' % myAlgorithm.passTime)
    #     print('非支配个体数：%s 个' % NDSet.sizes)



