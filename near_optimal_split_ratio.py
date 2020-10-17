# -*- coding: utf-8 -*-

from fractions import Fraction

import geatpy as ea
import numpy as np

max_val = float('inf')


class NearOptimalSplitRatioProblem(ea.Problem):
    """
    目标与父问题一致但计算的是针对以有向无环图中某一个节点v为目标节点的所有的链路的目标值
    .s.t sum(flow_split_ratio(v)) = 1
    """
    def __init__(self, dag=None, topological_sorted_nodes=None, traffic=None, sdn_nodes=None, band_width=None):
        """
        构造方法
        :param dag: 某一节点的有向无环图
        :param topological_sorted_nodes: 拓扑排序
        :param traffic: 流量需求,有向
        :param sdn_nodes: sdn节点列表
        :param band_width: 初始带宽，0代表本节点或节点间无直连链路，无向，用矩阵上半三角表示链路带宽
                           [[0,100,0]
                            [0, 0, 400]
                            [0, 0, 0]]
        """
        name = 'SplittingRatio'
        self.dag = dag
        self.topological_sorted_nodes = topological_sorted_nodes
        self.traffic = traffic
        self.sdn_nodes = sdn_nodes
        self.node_count = len(dag)
        self.sdn_node_count = len(sdn_nodes)
        self.band_width = band_width
        # 将每个sdn节点的分流数量计算存储为一个数组
        self.sdn_node_link_count = {}
        self.sdn_node_ratio_start_index = {}
        # 直连链路数量
        self.direct_link_count = np.count_nonzero(self.band_width)
        prev = 0
        for sdn_index in self.sdn_nodes:
            link_weight, link_count = np.unique(self.dag[sdn_index], return_counts=True)
            # 流量不转发给自己，记录有向无环图上该点所有的直连链路数量
            link_weight_count_dict = dict(zip(link_weight, link_count))
            max_weight_count = link_weight_count_dict[max_val] if max_val in link_weight_count_dict else 0
            sdn_node_link_count = self.node_count - 1 - max_weight_count
            # 如果在去往目标节点的链路上的sdn只有一条出链路，无需分流，特殊处理，减小个体染色体复杂度
            if sdn_node_link_count == 1 or sdn_node_link_count == 0:
                pass
            else:
                # 每个链路节点再分流染色体的起始索引
                self.sdn_node_link_count[sdn_index] = self.node_count - 1 - max_weight_count
                self.sdn_node_ratio_start_index[sdn_index] = prev
                prev += self.sdn_node_link_count[sdn_index]
        print(self.sdn_node_link_count)
        # 目标
        M = 2
        maxormins = [1, 1]
        Dim = sum(self.sdn_node_link_count.values())
        lb = [0] * Dim
        ub = [1] * Dim
        # 0:实数
        varTypes = [0] * Dim
        # 上下边界是否可以取到
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        """
        计算目标值函数
        :param pop: 种群
        :return:
        """
        ratio_matrix_pop = pop.Phen
        # 每个个体横向取值仿真打流获取该个体的目标函数的参数
        print(ratio_matrix_pop)
        obj_val = []
        for ratio_matrix in ratio_matrix_pop:
            link_band_width_used = self.route_flow(ratio_matrix)
            value_of_utilization_formula = self.calc_utilization_formula(link_band_width_used)
            # 这个需要计算有流量经过的链路的剩余带宽方差
            remaining_bandwidth_variance = self.calc_variance(link_band_width_used)
            obj_val.append([value_of_utilization_formula, remaining_bandwidth_variance])
        pop.ObjV = np.hstack([obj_val])
        # .s.t sum(flow_ratio(node_v)) = 1 可行性法则
        splitted_cv = []
        prev = 0
        for key in self.sdn_node_link_count.keys():
            splitted_cv\
                .append(abs(sum(ratio_matrix_pop[:, prev + i] for i in range(self.sdn_node_link_count[key])) - 1))
            prev += self.sdn_node_link_count[key]
        pop.CV = np.hstack([splitted_cv]).T

    def route_flow(self, ratio_matrix):
        # 内遗传算法中根据有向无环图打流时流量还是有向的，到父遗传中再合并流量转成无向
        link_band_width_used = np.zeros([self.node_count, self.node_count])
        # 拓扑排序最后一个节点为目标节点
        flow_demand_to_target = self.traffic[:, self.topological_sorted_nodes[-1]]
        # 记录已经经过的sdn节点数量
        # 遍历完拓扑排序，流量就已经全部到达目标节点
        for topo_node in self.topological_sorted_nodes:
            # 当前节点去往目标节点的初始流量需求，原始需求+上游流量需求
            flow_demand = flow_demand_to_target[topo_node] + sum(link_band_width_used[:, topo_node])
            next_hops = [i for i in self.dag[topo_node] if 0 < i < max_val]
            # 到达目标节点,结束
            if len(next_hops) == 0:
                continue
            # 决策变量在个体染色体中的位置
            split_ratio_index = self.sdn_node_ratio_start_index[topo_node] \
                if topo_node in self.sdn_nodes else 0
            for next_hop in next_hops:
                if topo_node in self.sdn_nodes and len(next_hops) != 1:
                    # 当前节点为sdn节点，按照生成的个体获取分流比例
                    link_band_width_used[topo_node][next_hop] = flow_demand * ratio_matrix[split_ratio_index]
                    split_ratio_index += 1
                else:
                    # 当前节点为普通节点，则按照ECMP规则流量等分
                    link_band_width_used[topo_node][next_hop] += flow_demand / len(next_hops)
        # 标量化,所有流量的带宽占用量都集中再矩阵的上半三角
        return np.triu(link_band_width_used) + np.tril(link_band_width_used).T

    def calc_variance(self, link_band_width_used):
        """

        计算链路剩余带宽方差
        :param link_band_width_used: 对一个目标节点打流之后的链路带宽使用情况
        :return:
        """
        remaining_bandwidth = self.band_width - link_band_width_used
        avg_remaining_bandwidth = np.sum(remaining_bandwidth) / self.direct_link_count
        variance_sum = 0
        for i in range(self.node_count):
            for j in range(self.node_count):
                # 判断一下两点之间是否有直接链接
                if self.band_width[i][j] != 0:
                    variance_sum += (remaining_bandwidth[i][j] - avg_remaining_bandwidth) ** 2
        return variance_sum / self.direct_link_count

    def calc_utilization_formula(self, link_band_width_used):
        utilization_matrix = link_band_width_used / self.band_width
        max_utilization = np.max(utilization_matrix)
        max_x_index, max_y_index = np.unravel_index(np.argmax(max_utilization), max_utilization.shape)
        max_utilization_bandwidth_used = link_band_width_used[max_x_index][max_y_index]
        max_utilization_raw_bandwidth = self.band_width[max_x_index][max_y_index]
        if 0 <= max_utilization <= Fraction(1, 3):
            return max_utilization_bandwidth_used
        elif Fraction(1, 3) < max_utilization <= Fraction(2, 3):
            return 3 * max_utilization_bandwidth_used - Fraction(2, 3) * max_utilization_raw_bandwidth
        elif Fraction(2, 3) < max_utilization <= Fraction(9, 10):
            return 10 * max_utilization_bandwidth_used - Fraction(16, 3) * max_utilization_raw_bandwidth
        elif Fraction(9, 10) < max_utilization <= 1:
            return 70 * max_utilization_bandwidth_used - Fraction(178, 3) * max_utilization_raw_bandwidth
        elif 1 < max_utilization <= Fraction(11, 10):
            return 500 * max_utilization_bandwidth_used - Fraction(1468, 3) * max_utilization_raw_bandwidth
        else:
            return 5000 * max_utilization_bandwidth_used - Fraction(16318, 3) * max_utilization_raw_bandwidth
