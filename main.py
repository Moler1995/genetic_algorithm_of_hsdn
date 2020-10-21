# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import geatpy as ea
import numpy as np
from hsdn_near_optimal_performance import SOHybridNetTEOptimizeProblem
max_val = float('inf')

if __name__ == '__main__':
    graph = np.array([[0, 1, 1, max_val], [1, 0, 1, 1], [1, 1, 0, 1], [max_val, 1, 1, 0]])
    band_width = np.array([[0.0, 100.0, 100.0, 0.0], [0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 0.0, 100.0], [0.0, 0.0, 0.0, 0.0]])
    traffic = np.array([[0, 10, 15, 10], [10, 0, 9, 8], [10, 10, 0, 20], [10, 10, 10, 0]])
    sdn_node_count = 1
    problem = SOHybridNetTEOptimizeProblem(graph, sdn_node_count, traffic, band_width)
    Encoding = ['P', 'RI']
    Field1 = ea.crtfld(Encoding[0], problem.varTypes[:sdn_node_count],
                       problem.ranges[:sdn_node_count], problem.borders[:sdn_node_count])  # 创建区域描述器
    Field2 = ea.crtfld(Encoding[1], problem.varTypes[sdn_node_count:],
                       problem.ranges[sdn_node_count:], problem.borders[sdn_node_count:])  # 创建区域描述器
