# -*- coding: utf-8 -*-
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import geatpy as ea
import numpy as np

import calculator

max_val = float('inf')
split_ratio_matrix = np.array([[None] * 12] * 12)
split_ratio_matrix[10][0] = [0.64, 0.36]
split_ratio_matrix[10][1] = [0, 1]
split_ratio_matrix[11][2] = [0.8156, 0.1844]
split_ratio_matrix[11][3] = [1, 0]
split_ratio_matrix[10][4] = [0, 1]
split_ratio_matrix[11][4] = [1, 0]
split_ratio_matrix[10][5] = [0, 1]
split_ratio_matrix[11][5] = [0.336, 0.664]
split_ratio_matrix[10][6] = [0, 1]


def execute(dag, topological_sorted_nodes, traffics, bandwidth, sdn_nodes, scene_determined_split_ratio=False,
            scene_verification=False):
    problem = NearOptimalSplitRatioProblem(dag=dag, topological_sorted_nodes=topological_sorted_nodes,
                                           sdn_nodes=sdn_nodes, band_width=bandwidth, traffics=traffics)
    if len(problem.sdn_node_link_count) == 0:
        return problem.parallel_route_flow_simulate(None)
    if scene_determined_split_ratio:
        ratio_matrix = __determined_split_ratio(dag, sdn_nodes, problem, bandwidth, scene_verification)
        return problem.parallel_route_flow_simulate(ratio_matrix)
    Encoding = "RI"
    NIND = 100
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ea.Population(Encoding, Field, NIND)
    myAlgorithm = ea.moea_NSGA3_DE_templet(problem, population)
    myAlgorithm.mutOper.F = 0.74
    myAlgorithm.recOper.XOVR = 0.65
    # 自定义初始种群,计算目标函数值和约束
    initChrom = __init_pre_chrom(NIND, sdn_nodes, problem)
    print('场景2：sdn节点最优分流比例进化算法开始....')
    unit_ratio = 1 / NIND
    prophetPop = ea.Population(Encoding, Field, NIND,
                               np.array(initChrom) * unit_ratio, Phen=np.array(initChrom) * unit_ratio)
    problem.aimFunc(prophetPop)
    myAlgorithm.MAXGEN = 70
    myAlgorithm.drawing = 1
    NDSet = myAlgorithm.run(prophetPop)
    return build_result_information(NDSet, problem, dag, topological_sorted_nodes[-1], True)


def __determined_split_ratio(dag, sdn_nodes, problem, bandwidth, scene_verify=False):
    ratio_matrix = []
    if scene_verify:
        target = problem.topological_sorted_nodes[-1]
        target_split_matrix = split_ratio_matrix[:, target]
        for split_ratio in target_split_matrix:
            if split_ratio:
                ratio_matrix += split_ratio
        return np.array(ratio_matrix)
    for sdn_node in sdn_nodes:
        if sdn_node not in problem.sdn_node_link_count.keys():
            continue
        node_total_bandwidth = 0
        for tg_node in range(len(dag)):
            if 0 < dag[sdn_node][tg_node] < max_val:
                node_total_bandwidth += bandwidth[sdn_node][tg_node] + bandwidth[tg_node][sdn_node]
        for tg_node in range(len(dag)):
            if 0 < dag[sdn_node][tg_node] < max_val:
                ratio_matrix.append((bandwidth[sdn_node][tg_node] + bandwidth[tg_node][sdn_node]) / node_total_bandwidth)
    return np.array(ratio_matrix)


def __init_pre_chrom(NIND, sdn_nodes, problem):
    """
    自定义初始种群,计算目标函数值和约束
    :param NIND:
    :param sdn_nodes:
    :param problem:
    :return:
    """
    initChrom = []
    chrom = []
    for j in range(0, NIND):
        start_index = 0
        for sdn_node in sdn_nodes:
            if sdn_node not in problem.sdn_node_link_count.keys():
                continue
            # 到这里直连链路数量一定是大于等于2的
            link_count = problem.sdn_node_link_count[sdn_node]
            for index in range(link_count):
                if index == link_count - 1:
                    chrom.append(NIND - sum(chrom[start_index:start_index + index]))
                    start_index += link_count
                else:
                    chrom.append(random.randint(0, NIND - sum(chrom[start_index:start_index + index])))
        if len(chrom) > 0:
            initChrom.append(chrom.copy())
            chrom.clear()
    return initChrom


def build_result_information(NDSet, problem, dag, target_node, do_print):
    """
    返回子问题多目标优化的近似最优解，难点：从进化算法得出的帕累托非支配解中选择最想要的点，(两个目标的权重选择策略)
    看优化重心在最小的最大链路利用率，还是最小剩余链路带宽方差
    :param NDSet: 解集
    :param problem: 问题实体
    :param dag: 有向无环图
    :param target_node: 目标节点
    :param do_print 是否打印结果
    :return: 最优解
    """
    # weight_ratio = sum(NDSet[:, 1] / NDSet[:, 0]) / len(NDSet)
    NDSet.save()
    optimal_solution_weight = [0, 1]
    weighted_NDSet = NDSet.ObjV[:, 0] * optimal_solution_weight[0] + NDSet.ObjV[:, 1] * optimal_solution_weight[1]
    near_optimal_bandwidth_used = problem.parallel_route_flow_simulate(NDSet.Phen[np.argmin(weighted_NDSet)])
    if not do_print:
        return near_optimal_bandwidth_used
    node_count = len(dag)
    near_opt_result_info = '加权后的解: \n'
    start_index = 0
    for sdn_node in problem.sdn_node_link_count.keys():
        length = problem.sdn_node_link_count[sdn_node]
        curr_sdn_node_info = 'sdn节点: %d, 目标节点: %d, 下一跳列表: %s, 加权后近似最优解: %s\n' \
                             % (sdn_node, target_node, [i for i in range(node_count) if 0 < dag[sdn_node][i] < max_val],
                                NDSet.Phen[np.argmin(weighted_NDSet)][start_index:start_index + length])
        start_index += length
        near_opt_result_info += curr_sdn_node_info
    print(near_opt_result_info)
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
    F2=σ(B(e)) = (Σ(B(e_i)-B(e))^2 / N(e)) ** (1/2)
    .s.t sum(flow_split_ratio(v)) = 1
    """
    def __init__(self, dag=None, topological_sorted_nodes=None, sdn_nodes=None, band_width=None, traffics=None):
        """
        构造方法
        :param dag: 某一节点的有向无环图
        :param topological_sorted_nodes: 拓扑排序
        :param sdn_nodes: sdn节点列表
        :param band_width: 初始带宽，0代表本节点或节点间无直连链路，无向，用矩阵上半三角表示链路带宽
                        e.g.[[0,100,0]
                            [0, 0, 400]
                            [0, 0, 0]]
        :param traffics: 历史流量矩阵
        """
        name = 'SplittingRatio'
        self.dag = dag
        self.topological_sorted_nodes = topological_sorted_nodes
        self.traffics = traffics
        self.sdn_nodes = sdn_nodes
        self.node_count = len(dag)
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
        self.generation = 0
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        """
        计算目标值函数
        :param pop: 种群
        :return:
        """
        ratio_matrix_pop = pop.Phen
        # 每个个体横向取值仿真打流获取该个体的目标函数的参数
        obj_val = []
        worker_count = 5
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            jobs = []
            for ratio_matrix in ratio_matrix_pop:
                jobs.append(executor.submit(self.get_target_values, ratio_matrix))
            for job in as_completed(jobs):
                utilization_formula_val, variance = job.result()
                obj_val.append([utilization_formula_val, variance])
        pop.ObjV = np.hstack([obj_val])
        # .s.t sum(flow_ratio(node_v)) = 1 可行性法则
        splitted_cv = []
        prev = 0
        for key in self.sdn_node_link_count.keys():
            splitted_cv\
                .append(abs(sum(ratio_matrix_pop[:, prev + i] for i in range(self.sdn_node_link_count[key])) - 1))
            prev += self.sdn_node_link_count[key]
        pop.CV = np.hstack([splitted_cv]).T
        print("end generation:", self.generation)
        self.generation += 1

    def get_target_values(self, ratio_matrix):
        link_band_width_used = self.parallel_route_flow_simulate(ratio_matrix)
        value_of_utilization_formula = calculator.calc_utilization_formula(self.band_width,
                                                                           link_band_width_used)
        # 这个需要计算有流量经过的链路的剩余带宽方差
        remaining_bandwidth_variance = calculator.calc_remaining_bandwidth_variance(self.band_width,
                                                                                    link_band_width_used)
        return value_of_utilization_formula, remaining_bandwidth_variance

    def parallel_route_flow_simulate(self, ratio_matrix):
        traffic_num = len(self.traffics)
        if traffic_num < 10000:
            return self.route_flow_traffics(ratio_matrix, 0, traffic_num) / traffic_num
        worker_count = 4
        step = int(traffic_num / worker_count)
        total_bandwidth_used = np.zeros([self.node_count, self.node_count])
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            jobs = []
            start = 0
            for index in range(worker_count):
                jobs.append(executor.submit(self.route_flow_traffics, ratio_matrix, start, step))
                start += step
            for job in as_completed(jobs):
                total_bandwidth_used = total_bandwidth_used + job.result()
        return total_bandwidth_used / traffic_num

    def route_flow_traffics(self, ratio_matrix, start, step):
        avg = np.zeros([self.node_count, self.node_count])
        traffic_slice = self.traffics[start:start + step] if start + step <= len(self.traffics) \
            else self.traffics[start:len(self.traffics)]
        for one_traffic in traffic_slice:
            avg += self.route_flow(ratio_matrix, one_traffic)
        return avg

    def route_flow(self, ratio_matrix, traffic):
        link_band_width_used = np.zeros([self.node_count, self.node_count])
        # 拓扑排序最后一个节点为目标节点
        flow_demand_to_target = traffic[:, self.topological_sorted_nodes[-1]]
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

