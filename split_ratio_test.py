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
    traffic = np.array([[0, 0, 0, 9], [0, 0, 0, 8], [0, 0, 0, 12], [0, 0, 0, 0]])
    bandwidth = np.array([[0, 10, 20, 0], [0, 0, 10, 30], [0, 0, 0, 15], [0, 0, 0, 0]])
    sdn_nodes = [1, 2]
    problem = NearOptimalSplitRatioProblem(dag=dag, topological_sorted_nodes=topological_sorted_nodes,
                                           traffic=traffic, sdn_nodes=sdn_nodes, band_width=bandwidth)
    Encoding = "RI"
    NIND = 50
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)
    myAlgorithm.MAXGEN = 200
    NDSet = myAlgorithm.run()
    NDSet.save()

