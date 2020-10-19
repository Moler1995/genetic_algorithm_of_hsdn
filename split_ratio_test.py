from collections import deque

import numpy as np
from near_optimal_split_ratio import NearOptimalSplitRatioProblem
import geatpy as ea

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
                                           traffic=traffic, sdn_nodes=sdn_nodes, band_width=bandwidth)
    Encoding = "RI"
    NIND = 100
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    # myAlgorithm = ea.moea_NSGA2_templet(problem, population) # 没算出结果，再看看
    myAlgorithm = ea.moea_NSGA3_DE_templet(problem, population) # 有结果，再分析看看
    # myAlgorithm = ea.moea_NSGA2_DE_templet(problem, population) # 有结果，再对比分析看看
    # myAlgorithm = ea.moea_MOEAD_DE_templet(problem, population) # 结果比较诡异，需要分析，执行时间长
    myAlgorithm.F = 0.75
    myAlgorithm.recOper.XOVR = 0.65
    # myAlgorithm = ea.moea_awGA_templet(problem, population) # 结果比较诡异，需要分析
    # myAlgorithm = ea.moea_MOEAD_archive_templet(problem, population) # 没结果
    # myAlgorithm = ea.moea_MOEAD_templet(problem, population) # 没算出结果，再看看
    # myAlgorithm = ea.moea_NSGA3_templet(problem, population) # 没算出结果，再看看
    myAlgorithm.MAXGEN = 100
    myAlgorithm.drawing = 2
    NDSet = myAlgorithm.run()
    NDSet.save()

