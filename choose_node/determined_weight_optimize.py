import geatpy as ea
import numpy as np
import graph_util as gu
from collections import deque
import near_optimal_split_ratio as nosr
import calculator
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


class NearOptLinkParamsWithDeterminedWeight(ea.Problem):
    def __init__(self, graph=None, sdn_node_count=0, traffic=None, band_width=None, xml_name=None):
        """
        构造函数
        :param graph: 连接图,有连接为1无连接为max_val,本节点为0
        :param sdn_node_count: 节点数量
        :param traffic: 流量需求图
        """
        name = 'EASOTE'
        node_size = len(graph) if graph is not None else RuntimeError("Graph can not be None")
        # 两维
        M = 2
        # 决策变量维数（长度为sdn节点数量）
        Dim = sdn_node_count
        # 目标函数求最大还是最小
        maxormins = [1, 1]
        # 决策变量类型1表示离散形变量
        varTypes = [1] * Dim
        # 决策变量下界,
        lb = [0] * sdn_node_count
        # 决策变量上界
        ub = [node_size - 1] * sdn_node_count
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
