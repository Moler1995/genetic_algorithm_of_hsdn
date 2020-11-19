# -*- coding: utf-8 -*-
import geatpy as ea
import numpy as np
import graph_util as gu
from collections import deque
import near_optimal_split_ratio as nosr
import calculator
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


'''
研究OSPF-SDN混合网络的ECMP权重优化，节点升级策略及SDN节点近似最优分流策略
目标1：为最小化的链路利用率函数指标
Φ(e)  -- l(e),               0<l(e)/c(e)<=1/3
       --3l(e)-2/3c(e),       1/3<l(e)/c(e)<=2/3
       --10l(e)-16/3c(e),     2/3<l(e)/c(e)<=9/10
       --70l(e)-178/3c(e),    9/10<l(e)/c(e)<=1
       --500l(e)-1468/3c(e)   1<l(e)/c(e)<=11/10
       --5000l(e)-16318/3c(e) l(e)/c(e)>11/10
目标2：最小化剩余带宽方差，在不同节点间的初始链路带宽不同的情况下，比如A:45/50和B:90/100的链路使用量，链路A和B的链路利用率相同，
但A链路仅剩5的带宽而B有10的剩余带宽，因此需要尝试减小链路之间的带宽差距，即缩小剩余带宽的方差来防止在链路有新增带宽需求因A和B的承
载能力差距而出现拥堵，但这一目标一定会以劣化目标函数1为代价
σ(B(e)) = Σ(B(e_i)-B(e))^2 / N(e)

本算法的最终目标是找到一组帕累托解，并定义目标1和目标2的权重，从这组解中选择出一个最优解，使最小化链路利用率和最小化带宽方差达到纳什均衡条件

约束条件
1.流量守恒
2.OSPF开销[1,64]∩N
3.SDN分流个数<=出链路个数
4.SDN分流比例之和为1(出节点流量守恒)
5.u(e)=l(e)/c(e)<=1
'''
max_val = float('inf')


class SOHybridNetTEOptimizeProblem(ea.Problem):
    def __init__(self, graph=None, sdn_node_count=0, traffic=None, band_width=None, xml_name=None):
        """
        构造函数
        :param graph: 连接图,有连接为1无连接为max_val,本节点为0
        :param sdn_node_count: 节点数量
        :param traffic: 流量需求图
        """
        name = 'SOEADETE'
        node_size = len(graph) if graph is not None else RuntimeError("Graph can not be None")
        weight_size = int(np.sum(graph == 1) / 2)
        # 两维
        M = 2
        # 决策变量维数（多目标多染色体编码，长度为sdn节点数量+权重数量）
        Dim = sdn_node_count + weight_size
        # 目标函数求最大还是最小
        maxormins = [1, 1]
        # 决策变量类型1表示离散形变量
        varTypes = [1] * Dim
        # 决策变量下界,从2开始是因为，sdn节点优化的时候会将sdn节点增加的出链路的权重改为1，且初始链路连接矩阵中用1表示有连接
        lb = [0] * sdn_node_count + [2] * weight_size
        # 决策变量上界,权重的搜索空间尽量低一点,降低搜索复杂度
        ub = [node_size - 1] * sdn_node_count + [weight_size + 1] * weight_size
        # 决策变量上下边界是否能取到
        lbin = [1] * Dim
        ubin = [1] * Dim
        # 父类构造
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 链路信息参数
        self.sdn_node_count = sdn_node_count
        self.node_size = node_size
        self.graph = graph
        self.traffic = traffic
        self.band_width = band_width
        self.xml_name = xml_name

    def aimFunc1(self, pop):
        pop_values = pop.Phen
        # 对种群中的每一个个体求目标值的近似最小值及解集
        obj_val_list = []
        for one_pop in pop_values:
            # 给连接图的上半三角填充权重
            sdn_nodes = np.array(one_pop[:self.sdn_node_count]).astype(int)
            filled_weight_list = self.fill_graph_weights(one_pop[self.sdn_node_count:])
            total_bandwidth_used = np.zeros([self.node_size, self.node_size])
            # Dijkstra算法对每个顶点计算最短链路
            shortest_path_list = [gu.dijkstra_alg(filled_weight_list, i) for i in range(self.node_size)]
            for i in range(self.node_size):
                # 先将每个节点看成传统节点，以每个顶点为目标节点，构建有向无环图
                legacy_node_dag = gu.build_dag(filled_weight_list, i, shortest_path_list)
                # 针对每一个顶点的有向无环图查找sdn节点，增加可用链路并验证环路
                dag, sorted_nodes = gu.add_links(filled_weight_list, legacy_node_dag, i, sdn_nodes)
                near_optimal_bandwidth_used = nosr.execute(dag, sorted_nodes, self.traffic, self.band_width, sdn_nodes,
                                                           scene_determined_split_ratio=True)
                # print('sdn节点为%s, %d为目标的, 近似最优链路使用情况:\n' % (sdn_nodes, i),
                #       near_optimal_bandwidth_used)
                total_bandwidth_used = near_optimal_bandwidth_used + total_bandwidth_used
            max_utilization_formula_val = calculator.calc_utilization_formula(self.band_width,
                                                                              total_bandwidth_used, False)
            min_variance = calculator.calc_remaining_bandwidth_variance(self.band_width, total_bandwidth_used)
            obj_val_list.append([max_utilization_formula_val, min_variance])
            max_utilization, max_x_index, max_y_index = calculator.calc_max_utilization(self.band_width, total_bandwidth_used)
            # print(self.xml_name + ": ", max_utilization)
            # print("target_one: " + str(max_utilization_formula_val) + " min_variance: " + str(min_variance))
            return max_utilization, max_x_index, max_y_index, max_utilization_formula_val, min_variance
            # print(total_bandwidth_used)
        # pop.ObjV = np.hstack(obj_val_list)

    def aimFunc(self, pop):
        pop_values = pop.Phen
        # 对种群中的每一个个体求目标值的近似最小值及解集
        pop_size = len(pop_values)
        obj_val_list = np.zeros([pop_size, self.M])
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=1) as executor:
            for index, result in zip(range(pop_size), executor.map(self.solve_one_pop, pop_values)):
                obj_val_list[index] = result
        # print('计算一个种群总耗时:{}'.format(time.time() - start_time))
        # print(obj_val_list)
        pop.ObjV = obj_val_list

    def solve_one_pop(self, one_pop):
        # 给连接图的上半三角填充权重
        start_time = time.time()
        sdn_nodes = np.array(one_pop[:self.sdn_node_count]).astype(int)
        filled_weight_list = self.fill_graph_weights(one_pop[self.sdn_node_count:])
        total_bandwidth_used = np.zeros([self.node_size, self.node_size])
        # Dijkstra算法对每个顶点计算最短链路
        shortest_path_list = [gu.dijkstra_alg(filled_weight_list, i) for i in range(self.node_size)]
        with ProcessPoolExecutor(max_workers=2) as executor:
            jobs = []
            for index in range(self.node_size):
                jobs.append(executor.submit(self.solve_sub_problem_one_node, index, filled_weight_list,
                                            shortest_path_list, sdn_nodes))
            for job in as_completed(jobs):
                total_bandwidth_used = total_bandwidth_used + job.result()
        min_utilization_formula_val = calculator.calc_utilization_formula(self.band_width, total_bandwidth_used, True)
        min_variance = calculator.calc_remaining_bandwidth_variance(self.band_width, total_bandwidth_used)
        max_utilization = calculator.calc_max_utilization(self.band_width, total_bandwidth_used)
        print(self.xml_name + ": ", max_utilization)
        print("target_one: " + str(min_utilization_formula_val) + " min_variance: " + str(min_variance))
        # print('计算一个个体的总耗时:{}'.format(time.time() - start_time))
        return [min_utilization_formula_val, min_variance]

    def solve_sub_problem_one_node(self, i, filled_weight_list, shortest_path_list, sdn_nodes):
        # 先将每个节点看成传统节点，以每个顶点为目标节点，构建有向无环图
        legacy_node_dag = gu.build_dag(filled_weight_list, i, shortest_path_list)
        # 针对每一个顶点的有向无环图查找sdn节点，增加可用链路并验证环路
        dag, sorted_nodes = gu.add_links(filled_weight_list, legacy_node_dag, i, sdn_nodes)
        near_optimal_bandwidth_used = nosr.execute(dag, sorted_nodes, self.traffic, self.band_width, sdn_nodes)
        # print('sdn节点为%s, %d为目标的, 近似最优链路使用情况:\n' % (sdn_nodes, i), near_optimal_bandwidth_used)
        return near_optimal_bandwidth_used

    def fill_graph_weights(self, weight_list):
        weight_list_tmp = deque(weight_list.copy())
        weight_to_fill = self.graph.copy()
        for i in range(self.node_size):
            for j in range(self.node_size):
                if i <= j:
                    continue
                else:
                    if weight_to_fill[i][j] < max_val:
                        weight_to_fill[i][j] = weight_list_tmp.popleft()
                        weight_to_fill[j][i] = weight_to_fill[i][j]
        return weight_to_fill
