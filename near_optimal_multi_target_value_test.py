import numpy as np
import geatpy as ea
from hsdn_near_optimal_performance import SOHybridNetTEOptimizeProblem
max_val = float('inf')


if __name__ == "__main__":
    # 初始化连接图
    graph = np.array([[0, 1, 1, max_val], [1, 0, 1, 1], [1, 1, 0, 1], [max_val, 1, 1, 0]])
    band_width = np.array([[0, 100, 100, 0], [0, 0, 100, 100], [0, 0, 0, 100], [0, 0, 0, 0]])
    traffic = np.array([[0, 10, 15, 10], [10, 0, 9, 8], [10, 10, 0, 20], [10, 10, 10, 0]])
    sdn_node_count = 1
    problem = SOHybridNetTEOptimizeProblem(graph, sdn_node_count, traffic, band_width)
    Encodings = ['P', 'RI']
    Field1 = ea.crtfld(Encodings[0], problem.varTypes[:sdn_node_count],
                       problem.ranges[:sdn_node_count], problem.borders[:sdn_node_count])  # 创建区域描述器
    Field2 = ea.crtfld(Encodings[1], problem.varTypes[sdn_node_count:],
                       problem.ranges[sdn_node_count:], problem.borders[sdn_node_count:])  # 创建区域描述器
    Fields = [Field1, Field2]
    # 种群规模
    NIND = 100
    population = ea.PsyPopulation(Encodings, Fields, NIND)
    myAlgorithm = ea.moea_psy_NSGA3_templet(problem, population)
    myAlgorithm.MAXGEN = 200

    myAlgorithm.drawing = 2
    NDSet = myAlgorithm.run()
    NDSet.save()  # 把结果保存到文件中
    # 输出
    print('用时：%s 秒' % myAlgorithm.passTime)
    print('非支配个体数：%s 个' % NDSet.sizes)
    print('单位时间找到帕累托前沿点个数：%s 个' % (int(NDSet.sizes // myAlgorithm.passTime)))
