from collections import deque

import numpy as np

max_val = float('inf')


def dijkstra_alg(weight_matrix, node_index):
    """
    dijkstra单源最短路径，算得以该节点为目标节点的所有路由的最小cost
    :param weight_matrix: cost矩阵
    :param node_index: 待求最短路径的index
    :return:
    """
    node_count = len(weight_matrix)
    shortest_path_vals = [max_val] * node_count
    shortest_path_vals[node_index] = 0
    node_visited = [False] * node_count
    # 记录最短路径
    shortest_path = []
    next_index = node_index
    for i in range(node_count):
        node_visited[next_index] = True
        shortest_path.append(next_index)
        curr_weights = weight_matrix[next_index]
        curr_node_weight = shortest_path_vals[next_index]
        for weight_index in range(node_count):
            if weight_index == next_index:
                continue
            new_weight = curr_weights[weight_index] + curr_node_weight
            # 如果开销小于已经存储的值则更新
            if new_weight < shortest_path_vals[weight_index]:
                shortest_path_vals[weight_index] = new_weight
            if not node_visited[weight_index]:
                if node_visited[next_index]:
                    next_index = weight_index
                elif new_weight < shortest_path_vals[next_index]:
                    next_index = weight_index

    return shortest_path_vals


def find_all_shortest_paths(graph, source, target, min_cost):
    """
    根据dijkstra算到的源节点到目的节点的最短路径值，获得所有的最短路径
    :param graph: list 图的邻接矩阵
    :param source: int index
    :param target: int index
    :param min_cost: int cost
    :return:
    """
    # 用于存储所有的路径
    path_list = deque()
    # 临时路径存储列表，初始化并存入源节点
    path = deque([source])
    # 节点下一跳存储列表
    next_hops_list = deque()
    # 初始化源节点下一跳列表
    next_hops = __build_next_hops(graph, source, path, min_cost)
    if len(next_hops) == 0:
        return None
    next_hops_list.append(next_hops)
    # 临时路径列表非空
    while len(path) != 0:
        next_hops = next_hops_list.pop()
        if len(next_hops) == 0:
            path.pop()
            continue
        else:
            next_hop = next_hops.pop()
            next_hops_list.append(next_hops)
            path.append(next_hop)
            next_hops_list.append(__build_next_hops(graph, next_hop, path, min_cost))
            if next_hop == target:
                path_list.append(path.copy())
                path.pop()
                next_hops_list.pop()
    return path_list


def __build_next_hops(graph, node_index, path, min_cost):
    next_hops = deque()
    curr_cost = 0
    for index in range(len(path) - 1):
        curr_cost += graph[path[index]][path[index + 1]]
    for i in range(len(graph)):
        if i == node_index or path.__contains__(i):
            continue
        if curr_cost + graph[node_index][i] <= min_cost:
            next_hops.append(i)
    return next_hops


def build_dag(graph, source_index, min_weight_matrix):
    """
    构建以source_index节点为目标节点的有向无环图
    :param graph: 图
    :param source_index: 目标节点
    :param min_weight_matrix: 最小cost矩阵
    :return:
    """
    node_count = len(graph)
    dag = np.ones([node_count, node_count]) * max_val
    # 对角线置零，暂缓，为方便后面增加链路判断
    # dag[range(node_count), range(node_count)] = 0
    # source_index一开始作为源节点，计算出链路的有向无环图，最后将矩阵转置一下变为入链路，到最后入参source_index实际为target_index
    for target_index in range(node_count):
        if source_index == target_index:
            continue
        shortest_paths = find_all_shortest_paths(graph, source_index, target_index,
                                                 min_weight_matrix[source_index][target_index])
        for shortest_path in shortest_paths:
            if len(shortest_path) == 0:
                continue
            next_hop = shortest_path.popleft()
            while shortest_path.__len__() != 0:
                next_next_hop = shortest_path.popleft()
                if dag[next_next_hop][next_hop] == 1:
                    continue
                dag[next_hop][next_next_hop] = graph[next_hop][next_next_hop]
                next_hop = next_next_hop
    # 矩阵转置，变为节点的入链路的有向无环图
    return dag.T


def add_links(graph, dag, target_node, sdn_nodes):
    node_count = len(graph)
    for sdn_node in sdn_nodes:
        # 如果sdn节点为目标节点，就不会产生新的出链路
        if sdn_node == target_node:
            continue
        for dag_node_index in range(node_count):
            # sdn节点与另一节点在有向无环图里既没有出链路也没有入链路，且在原始的链路图中有链接，需要加上一条出链路并检查环路，
            # 并且先不管对端为sdn节点的情况
            if dag[sdn_node][dag_node_index] == max_val and dag[dag_node_index][sdn_node] == max_val \
                    and max_val > graph[sdn_node][dag_node_index] > 0 and dag_node_index not in sdn_nodes:
                # 给这条链路一个权重标识这条链路的通路，在进行流量转发的时候再根据分流比例种群定义的比例转发流量。
                dag[sdn_node][dag_node_index] = 1
                # 无法完成全拓扑排序，即存在环路，需要回退
                if len(topological_sort(dag)) < node_count:
                    dag[sdn_node][dag_node_index] = max_val
    # 最后对sdn节点对进行出链路补偿,有可能两个节点间任何方向加链路都没有环路，但只能选一个
    sdn_count = len(sdn_nodes)
    for sdn_node_index1 in range(sdn_count):
        for sdn_node_index2 in range(sdn_node_index1 + 1, sdn_count):
            node_index1 = sdn_nodes[sdn_node_index1]
            node_index2 = sdn_nodes[sdn_node_index2]
            if dag[node_index1][node_index2] == max_val and dag[node_index2][node_index1] == max_val \
                    and max_val > graph[node_index1][node_index2] > 0:
                # 如果index1是目标节点，则直接将链路设置为2->1并校验
                if node_index1 == target_node:
                    dag[node_index2][node_index1] = 1
                    if len(topological_sort(dag)) < node_count:
                        dag[node_index2][node_index1] = max_val
                    continue
                # 否则先由1->2，校验然后检测环路
                dag[node_index1][node_index2] = 1
                # 无法完成全拓扑排序，即存在环路，需要回退
                if len(topological_sort(dag)) == node_count:
                    continue
                else:
                    dag[node_index1][node_index2] = max_val
                    dag[node_index2][node_index1] = 1
                    if len(topological_sort(dag)) < node_count:
                        dag[node_index2][node_index1] = max_val
    # 对角线置0
    sorted_nodes = topological_sort(dag)
    dag[range(node_count), range(node_count)] = 0
    return dag, sorted_nodes


def topological_sort(dag):
    """
    对有向无环图进行拓扑排序，如果排序后的节点长度小于总长度，则说明存在环路
    :param dag: 有向无环图
    :return: list
    """
    tmp_dag = dag.copy()
    node_count = len(dag)
    sorted_nodes = deque()
    i = 0
    while i < node_count:
        if i not in sorted_nodes and np.all(tmp_dag[:, i] == max_val):
            sorted_nodes.append(i)
            tmp_dag[i] = max_val
            i = 0
        else:
            i += 1
    return sorted_nodes


def sss(a, b, c, d):
    print(a, b, c, d)
    return a


if __name__ == '__main__':
    graph = [[0, 2, 9, max_val], [2, 0, 7, 3], [9, 7, 0, 4], [max_val, 3, 4, 0]]
    # a, b = np.unravel_index(np.argmax(graph_arr), graph_arr.shape) # 计算最大值坐标
    # graph1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    # graph1 = graph1 + graph_arr
    # print(graph1)
    # shortest_path_list = [dijkstra_alg(graph, i) for i in range(4)]
    # print(shortest_path_list)
    # dag_list = [build_dag(graph, i, shortest_path_list) for i in range(4)]
    # dag, sorted_nodes = add_links(graph, dag_list[3], 3, [2, 1])
    # print(dag, sorted_nodes)
    import random
    from concurrent.futures import ProcessPoolExecutor, as_completed
    length = 5
    x = np.ones([length, length])
    with ProcessPoolExecutor(max_workers=4) as executor:
        for index, res in zip(range(10), executor.map(sss, x, x, x, x)):
            print('index:{}, res:{}'.format(index, res))
        # jobs = []
        # for i in range(4):
        #     jobs.append(executor.submit(sss, i, i+1, [i+2], [i+3]))
        # for job in as_completed(jobs):
        #     print(job.result())
    # ratio_matrix_pop = np.array([[0.1, 0.3, 0.6, 0.1, 0.9, 0.3, 0.9],
    #                              [0.1, 0.3, 0.4, 0.9, 0.7, 0.2, 0.5],
    #                              [0.3, 0.8, 0.2, 0.0, 1.0, 0.6, 0.5]])
    # sdn_nodes = [1, 2, 3]
    # sdn_node_link_count = {1: 3, 2: 2, 3: 2}
    # ratio_matrix_dict = {}
    # prev = 0
    # for sdn_index in sdn_nodes:
    #     # link_weight, link_count = np.unique(self.dag[sdn_index])
    #     ratio_matrix_dict[sdn_index] = ratio_matrix_pop[:, prev:prev + sdn_node_link_count[sdn_index]]
    #     prev += sdn_node_link_count[sdn_index]
    # print(ratio_matrix_dict)
    # splitted_cv = []
    # prev = 0
    # for key in sdn_node_link_count.keys():
    #     # split_matrix = ratio_matrix_dict[key]
    #     splitted_cv.append(abs(sum(ratio_matrix_pop[:, prev + i] for i in range(sdn_node_link_count[key])) - 1))
    #     prev += sdn_node_link_count[key]
    # print(splitted_cv)
    # print(np.hstack([splitted_cv]).T)
        # for i in range()
        # splitted_cv.append([abs(sum(split_matrix[i])) for i in range(sdn_node_link_count[key])])
    # print(np.array(splitted_cv))
    # a = np.ones([3, 10])
    # print(a)
    # print([sum(a[:, i] for i in range(len(a[0])))])







