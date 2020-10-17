# -*- coding: utf-8 -*-
import geatpy as ea

'''
研究OSPF-SDN混合网络的ECMP权重优化，节点升级策略及SDN节点近似最优分流策略
目标为最小化的链路利用率指标以及最小化剩余带宽方差
Φ(e)  -- l(e),            0<l(e)/c(e)<=1/3
       --3l(e)-2/3c(e),    1/3<l(e)/c(e)<=2/3
       --10l(e)-16/3c(e),  2/3<l(e)/c(e)<=9/10
       --70l(e)-178/3c(e), 9/10<l(e)/c(e)<=1
            _________________
σ(B(e)) = √Σ(B(e_i)-B(e))^2
约束条件
1.流量守恒
2.OSPF开销[1,64]∩N
3.SDN分流个数<=出链路个数
4.SDN分流比例之和为1(出节点流量守恒)
5.u(e)=l(e)/c(e)<=1
'''
max_val = float('inf')


class SOHybridNetTEOptimizeProblem(ea.Problem):
    def __init__(self, graph=None, sdn_node_count=0, traffic=None):
        name = 'SOAGATE'
        node_size = len(graph) if graph is not None else RuntimeError("Graph can not be None")
        weight_size = 0
        for node_line in graph:
            for node in node_line:
                if node < max_val:
                    weight_size += 1

        # 两维
        M = 2
        # 决策变量维数
        Dim = sdn_node_count + weight_size
        # 目标函数求最大还是最小
        maxormins = [1, 1]
        # 决策变量类型
        varTypes = [1] * Dim
        # 决策变量下界
        lb = [2] * Dim
        # 决策变量上界
        ub = [64] * weight_size + [node_size] * sdn_node_count
        # 决策变量上边界
        lbin = [1] * Dim
        ubin = [1] * Dim
        # 父类构造
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 链路信息参数
        self.weight_size = weight_size
        self.sdn_node_count = sdn_node_count
        self.node_size = node_size
        self.graph = graph
        self.traffic = traffic

    def aimFunc(self, pop):
        pop_values = pop.Phen
        # 对种群中的每一个个体求目标值的近似最小值及解集
        '''
        Dijkstra算法对每个顶点计算最短链路
        '''
        for one_pop in pop_values:
            pass


