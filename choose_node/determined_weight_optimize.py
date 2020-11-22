import geatpy as ea
import numpy as np
import graph_util as gu
import near_optimal_split_ratio as nosr
import calculator
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


class NearOptUpgradeStrategyWithDeterminedWeight(ea.Problem):
    def __init__(self, graph=None, traffics=None, band_width=None):
        """
        构造函数
        :param graph: 连接图,有连接为1无连接为max_val,本节点为0
        :param traffics: 流量需求图
        """
        name = 'EASOTE'
        node_size = len(graph) if graph is not None else RuntimeError("Graph can not be None")
        # 两维
        M = 2
        sdn_node_count = len(graph)
        # 决策变量维数（长度为sdn节点数量）
        Dim = sdn_node_count
        # 目标函数求最大还是最小
        maxormins = [1, 1]
        # 决策变量类型1表示离散形变量
        varTypes = [1] * Dim
        # 决策变量下界,
        lb = [0] * Dim
        # 决策变量上界
        ub = [node_size - 1] * Dim
        # 决策变量上下边界是否能取到
        lbin = [1] * Dim
        ubin = [1] * Dim
        # 父类构造
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 链路信息参数
        self.sdn_node_count = sdn_node_count
        self.node_count = node_size
        self.graph = graph
        self.traffics = traffics
        self.band_width = band_width

    def aimFunc(self, pop):
        pop_values = pop.Phen
        # 对种群中的每一个个体求目标值的近似最小值及解集
        pop_size = len(pop_values)
        obj_val_list = np.zeros([pop_size, self.M])
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=8) as executor:
            for index, result in zip(range(pop_size), executor.map(self.solve_one_pop, pop_values)):
                obj_val_list[index] = result
        # print('计算一个种群总耗时:{}'.format(time.time() - start_time))
        # print(obj_val_list)
        pop.ObjV = obj_val_list

    def solve_one_pop(self, one_pop, do_print=False):
        """
        尝试仿真流量计算链路的最大利用率函数和剩余带宽均方差的均值
        """
        # 给连接图的上半三角填充权重
        start_time = time.time()
        sdn_node_perm = np.array(one_pop).astype(int)
        print("sdn node permutation: ", sdn_node_perm)
        filled_weight_list = self.graph.copy()
        shortest_path_list = [gu.dijkstra_alg(filled_weight_list, i) for i in range(self.node_count)]
        traffic_count = len(self.traffics)
        total_mean_utilization_formula = 0
        total_mean_variance = 0
        total_mean_max_utilization = 0
        for traffic in self.traffics:
            mean_traffic_formula_val = 0
            mean_variance = 0
            mean_max_utilization = 0
            for i in range(self.node_count):
                # 计算渐进升级策略升级过程中应对历史数据的链路性能平均表现
                sdn_nodes = sdn_node_perm[:i]
                total_bandwidth_used = np.zeros([self.node_count, self.node_count])
                # Dijkstra算法对每个顶点计算最短链路
                for index in range(self.node_count):
                    total_bandwidth_used += self.solve_sub_problem_one_node(index, filled_weight_list,
                                                                            shortest_path_list, sdn_nodes, traffic)
                # with ProcessPoolExecutor(max_workers=2) as executor:
                #     jobs = []
                #     for index in range(self.node_count):
                #         jobs.append(executor.submit(self.solve_sub_problem_one_node, index, filled_weight_list,
                #                                     shortest_path_list, sdn_nodes, traffic))
                #     for job in as_completed(jobs):
                #         total_bandwidth_used = total_bandwidth_used + job.result()
                utilization_formula_val = calculator.calc_utilization_formula(self.band_width,
                                                                              total_bandwidth_used, do_print)
                variance = calculator.calc_remaining_bandwidth_variance(self.band_width, total_bandwidth_used)
                max_utilization = calculator.calc_max_utilization(self.band_width, total_bandwidth_used)[0]
                # print("sdn nodes: ", sdn_nodes, ": ", max_utilization)
                # print("target_one: " + str(utilization_formula_val) + " min_variance: " + str(variance))
                mean_traffic_formula_val += utilization_formula_val / self.node_count
                mean_variance += variance / self.node_count
                mean_max_utilization += max_utilization / self.node_count
            total_mean_utilization_formula += mean_traffic_formula_val / traffic_count
            total_mean_variance += mean_variance / traffic_count
            total_mean_max_utilization += mean_max_utilization / traffic_count
        print("sdn_node_permutation: ", sdn_node_perm, "result: ", (total_mean_utilization_formula, total_mean_variance,
                                                                    total_mean_max_utilization))
        # print('计算一个个体的总耗时:{}'.format(time.time() - start_time))
        return [total_mean_utilization_formula, total_mean_variance]

    def solve_sub_problem_one_node(self, i, filled_weight_list, shortest_path_list, sdn_nodes, traffic):
        # 先将每个节点看成传统节点，以每个顶点为目标节点，构建有向无环图
        legacy_node_dag = gu.build_dag(filled_weight_list, i, shortest_path_list)
        # 针对每一个顶点的有向无环图查找sdn节点，增加可用链路并验证环路
        dag, sorted_nodes = gu.add_links(filled_weight_list, legacy_node_dag, i, sdn_nodes)
        near_optimal_bandwidth_used = nosr.execute(dag, sorted_nodes, traffic, self.band_width, sdn_nodes, True)
        # print('sdn节点为%s, %d为目标的, 近似最优链路使用情况:\n' % (sdn_nodes, i), near_optimal_bandwidth_used)
        return near_optimal_bandwidth_used

