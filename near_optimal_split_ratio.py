# -*- coding: utf-8 -*-
import geatpy as ea
import numpy as np
import random
import calculator

max_val = float('inf')


def execute(dag, topological_sorted_nodes, traffic, bandwidth, sdn_nodes):
    problem = NearOptimalSplitRatioProblem(dag=dag, topological_sorted_nodes=topological_sorted_nodes,
                                           traffic=traffic, sdn_nodes=sdn_nodes, band_width=bandwidth)
    Encoding = "RI"
    NIND = 100
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm = ea.moea_NSGA3_DE_templet(problem, population)
    myAlgorithm.mutOper.F = 0.74
    myAlgorithm.recOper.XOVR = 0.65
    # 自定义初始种群,计算目标函数值和约束
    initChrom = []
    unit_ratio = 1 / NIND
    chrom = []
    for j in range(0, NIND):
        start_index = 0
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
    if len(initChrom) == 0:
        # 所有sdn节点都只有一条出口链路，直接根据ecmp规则仿真打流，并计算此时的链路利用情况
        return problem.route_flow(None)
    else:
        prophetPop = ea.Population(Encoding, Field, NIND,
                                   np.array(initChrom) * unit_ratio, Phen=np.array(initChrom) * unit_ratio)
        problem.aimFunc(prophetPop)
        myAlgorithm.MAXGEN = 100
        myAlgorithm.drawing = 2
        NDSet = myAlgorithm.run(prophetPop)
        print('用时：%s 秒' % myAlgorithm.passTime)
        print('非支配个体数：%s 个' % NDSet.sizes)
        # 返回子问题多目标优化的近似最优解，难点：从进化算法得出的帕累托非支配解中选择最想要的点，(两个目标的权重选择策略)
        # 看优化重心在最小的最大链路利用率还是最小剩余链路带宽方差
        optimal_solution_weight = [0.4, 0.6]
        NDSet.ObjV[:, 0] + NDSet.ObjV[:, 1]
        near_optimal_bandwidth_used = []
        return near_optimal_bandwidth_used


class NearOptimalSplitRatioProblem(ea.Problem):
    """
    目标与父问题一致但计算的是针对以有向无环图中某一个节点v为目标节点的所有的链路的目标值
    F1=Φ(e)  -- l(e),        0<l(e)/c(e)<=1/3
       --3l(e)-2/3c(e),       1/3<l(e)/c(e)<=2/3
       --10l(e)-16/3c(e),     2/3<l(e)/c(e)<=9/10
       --70l(e)-178/3c(e),    9/10<l(e)/c(e)<=1
       --500l(e)-1468/3c(e)   1<l(e)/c(e)<=11/10
       --5000l(e)-16318/3c(e) l(e)/c(e)>11/10
    F2=σ(B(e)) = Σ(B(e_i)-B(e))^2 / N(e)
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
                        e.g.[[0,100,0]
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
        # 将每个sdn节点的分流数量计算存储为一个字典
        self.sdn_node_link_count = {}
        self.sdn_node_ratio_start_index = {}
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
        # print(ratio_matrix_pop)
        # 每个个体横向取值仿真打流获取该个体的目标函数的参数
        obj_val = []
        for ratio_matrix in ratio_matrix_pop:
            # todo: 这里可以考虑并行计算
            link_band_width_used = self.route_flow(ratio_matrix)
            value_of_utilization_formula = calculator.calc_utilization_formula(self.band_width, link_band_width_used)
            # 这个需要计算有流量经过的链路的剩余带宽方差
            remaining_bandwidth_variance = calculator.calc_remaining_bandwidth_variance(self.band_width,
                                                                                        link_band_width_used)
            obj_val.append([value_of_utilization_formula, remaining_bandwidth_variance])
        # print(obj_val)
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
        link_band_width_used = np.zeros([self.node_count, self.node_count])
        # 拓扑排序最后一个节点为目标节点
        flow_demand_to_target = self.traffic[:, self.topological_sorted_nodes[-1]]
        # 记录已经经过的sdn节点数量
        # 遍历完拓扑排序，流量就已经全部到达目标节点
        for topo_node in self.topological_sorted_nodes:
            # 当前节点去往目标节点的初始流量需求，原始需求+上游流量需求
            flow_demand = flow_demand_to_target[topo_node] + sum(link_band_width_used[:, topo_node])
            next_hops = [i for i in range(self.node_count) if 0 < self.dag[topo_node][i] < max_val]
            # 到达目标节点,结束
            if len(next_hops) == 0:
                continue
            # 决策变量在个体染色体中的位置
            split_ratio_index = self.sdn_node_ratio_start_index[topo_node] \
                if topo_node in self.sdn_nodes and topo_node in self.sdn_node_ratio_start_index.keys() else 0
            for next_hop in next_hops:
                if topo_node in self.sdn_nodes and len(next_hops) != 1:
                    # 当前节点为sdn节点，按照生成的个体获取分流比例
                    link_band_width_used[topo_node][next_hop] = flow_demand * ratio_matrix[split_ratio_index]
                    split_ratio_index += 1
                else:
                    # 当前节点为普通节点或是单条出链路的sdn节点，则按照ECMP规则流量等分
                    link_band_width_used[topo_node][next_hop] += flow_demand / len(next_hops)
        # 标量化带宽占用量,所有流量的带宽占用量都集中再矩阵的上半三角
        return np.triu(link_band_width_used) + np.tril(link_band_width_used).T

