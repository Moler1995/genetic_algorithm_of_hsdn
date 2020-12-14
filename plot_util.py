import matplotlib.pyplot as plt
import json
import os

color_warehouse = ['blue', 'green', 'yellow']


def plot_individual_utilization_result(month_index, threshold=0.0):
    json_file = "utilization/add_weight/abilene_TM_2004_%s.json" % month_index
    f = open(json_file, 'r', encoding="utf-8")
    result_dict = json.load(f, object_hook=dict)
    f.close()
    x_data = []
    y_data = []
    for key in result_dict.keys():
        if result_dict[key] <= threshold:
            continue
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y_data.append(result_dict[key])
    plt.figure(figsize=(15, 5))
    plt.plot(x_data, y_data, linewidth=0.6)
    plt.xticks([])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("%s月链路最大利用率大于%s的变化折线图" % (month_index, threshold))
    plt.xlabel("时间")
    plt.ylabel("链路最大利用率")
    plt.ylim((0, 1))
    # plt.savefig("./charts/individual/utilization/origin/%s.png" % month_index)
    plt.show()


def plot_utilization_compared_result(month_index, optimize_result_count, threshold=0.0):
    json_file_origin = "utilization/add_weight/abilene_TM_2004_%s.json" % month_index
    optimized_results = []
    for i in range(optimize_result_count):
        suffix = '' if i == 0 else 's'
        json_file_optimized = "utilization/upgrade_strategy_%d_node%s/abilene_TM_2004_%s.json" \
                              % ((1 + i), suffix, month_index)
        f = open(json_file_optimized, 'r', encoding='utf-8')
        optimized_result_dict = json.load(f, object_hook=dict)
        f.close()
        optimized_results.append(optimized_result_dict)
    f_1 = open(json_file_origin, 'r', encoding="utf-8")
    origin_result_dict = json.load(f_1, object_hook=dict)
    f_1.close()
    x_data = []
    y1_data = []
    y_optimized_data_dict = {}
    for key in origin_result_dict.keys():
        if origin_result_dict[key] <= threshold:
            continue
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y1_data.append(origin_result_dict[key])
        for i in range(optimize_result_count):
            if i in y_optimized_data_dict.keys():
                y_optimized_data_dict[i].append(optimized_results[i][key])
            else:
                y_optimized_data_dict[i] = [optimized_results[i][key]]
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x_data, y1_data, 'r', label='原始利用率', linewidth=0.6)
    for i in range(optimize_result_count):
        label_name = "升级%d个节点" % (i + 1)
        plt.plot(x_data, y_optimized_data_dict[i], color_warehouse[i], label=label_name, linewidth=0.8)
    plt.xticks([])
    plt.title("%s月链路利用率变化折线图" % month_index)
    plt.xlabel("时间")
    plt.ylabel("链路最大利用率")
    plt.ylim((0, 1))
    plt.legend(fontsize=10)
    plt.savefig("./charts/compare/utilization/compare_%s_upgraded/%s.png" % (optimize_result_count, month_index))
    plt.show()


def plot_utilization_func_val_chart(month_index, threshold=0.0):
    json_file_utilization_origin = "utilization/add_weight/abilene_TM_2004_%s.json" % month_index
    f_0 = open(json_file_utilization_origin)
    origin_utilization_dict = json.load(f_0, object_hook=dict)
    f_0.close()
    json_file = "utilization_function_value/add_weight/abilene_TM_2004_%s.json" % month_index
    f = open(json_file, 'r', encoding="utf-8")
    result_dict = json.load(f, object_hook=dict)
    f.close()
    x_data = []
    y_data = []
    for key in result_dict.keys():
        if origin_utilization_dict[key] <= threshold:
            continue
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y_data.append(result_dict[key])
    plt.figure(figsize=(15, 5))
    plt.plot(x_data, y_data, linewidth=0.6)
    plt.xticks([])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("%s月链路利用率函数变化折线图" % month_index)
    plt.xlabel("时间")
    plt.ylabel("链路利用率函数值")
    plt.savefig("./charts/individual/utilization_func_val/origin/%s.png" % month_index)
    plt.show()


def plot_utilization_func_val_compared_result(month_index, optimize_result_count, threshold=0.0):
    json_file_utilization_origin = "utilization/add_weight/abilene_TM_2004_%s.json" % month_index
    f_0 = open(json_file_utilization_origin)
    origin_utilization_dict = json.load(f_0, object_hook=dict)
    f_0.close()
    json_file_origin = "utilization_function_value/add_weight/abilene_TM_2004_%s.json" % month_index
    optimized_results = []
    for i in range(optimize_result_count):
        suffix = '' if i == 0 else 's'
        json_file_optimized = "utilization_function_value/upgrade_strategy_%d_node%s/abilene_TM_2004_%s.json"\
                              % ((1 + i), suffix, month_index)
        f = open(json_file_optimized, 'r', encoding='utf-8')
        optimized_result_dict = json.load(f, object_hook=dict)
        f.close()
        optimized_results.append(optimized_result_dict)
    f_1 = open(json_file_origin, 'r', encoding="utf-8")
    origin_result_dict = json.load(f_1, object_hook=dict)
    f_1.close()
    x_data = []
    y1_data = []
    y_optimized_data_dict = {}
    for key in origin_result_dict.keys():
        if origin_utilization_dict[key] <= threshold:
            continue
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y1_data.append(origin_result_dict[key])
        for i in range(optimize_result_count):
            if i in y_optimized_data_dict.keys():
                y_optimized_data_dict[i].append(optimized_results[i][key])
            else:
                y_optimized_data_dict[i] = [optimized_results[i][key]]
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    if optimize_result_count == 1:
        plt.plot(x_data, y1_data, 'r', label='原始利用率函数值', linewidth=0.6)
    for i in range(optimize_result_count):
        label_name = "升级%d个节点" % (i + 1)
        plt.plot(x_data, y_optimized_data_dict[i], color_warehouse[i], label=label_name, linewidth=0.6)
    plt.xticks([])
    plt.title("%s月链路利用率函数值变化折线图" % month_index)
    plt.xlabel("时间")
    plt.ylabel("链路利用率函数值")
    plt.legend(fontsize=10)
    plt.savefig("./charts/compare/utilization_func_val/compare_%s_upgraded/%s.png" % (optimize_result_count, month_index))
    plt.show()


def plot_variance_chart(month_index):
    json_file = "variance/add_weight/abilene_TM_2004_%s.json" % month_index
    f = open(json_file, 'r', encoding="utf-8")
    result_dict = json.load(f, object_hook=dict)
    f.close()
    x_data = []
    y_data = []
    for key in result_dict.keys():
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y_data.append(result_dict[key])
    plt.figure(figsize=(15, 5))
    plt.plot(x_data, y_data, linewidth=0.6)
    plt.xticks([])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("%s月链路剩余带宽标准差折线图" % month_index)
    plt.xlabel("时间")
    plt.ylabel("剩余带宽标准差")
    plt.savefig("./charts/individual/variance/origin/%s.png" % month_index)
    plt.show()


def plot_variance_compared_result(month_index, optimize_result_count, threshold=0.0):
    json_file_utilization_origin = "utilization/add_weight/abilene_TM_2004_%s.json" % month_index
    f_0 = open(json_file_utilization_origin)
    origin_utilization_dict = json.load(f_0, object_hook=dict)
    f_0.close()
    json_file_origin = "variance/add_weight/abilene_TM_2004_%s.json" % month_index
    optimized_results = []
    for i in range(optimize_result_count):
        suffix = '' if i == 0 else 's'
        json_file_optimized = "variance/upgrade_strategy_%d_node%s/abilene_TM_2004_%s.json"\
                              % ((1 + i), suffix, month_index)
        f = open(json_file_optimized, 'r', encoding='utf-8')
        optimized_result_dict = json.load(f, object_hook=dict)
        f.close()
        optimized_results.append(optimized_result_dict)
    f_1 = open(json_file_origin, 'r', encoding="utf-8")
    origin_result_dict = json.load(f_1, object_hook=dict)
    f_1.close()
    x_data = []
    y1_data = []
    y_optimized_data_dict = {}
    for key in origin_result_dict.keys():
        if origin_utilization_dict[key] <= threshold:
            continue
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y1_data.append(origin_result_dict[key])
        for i in range(optimize_result_count):
            if i in y_optimized_data_dict.keys():
                y_optimized_data_dict[i].append(optimized_results[i][key])
            else:
                y_optimized_data_dict[i] = [optimized_results[i][key]]
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x_data, y1_data, 'r', label='原始剩余带宽标准差', linewidth=0.6)
    for i in range(optimize_result_count):
        label_name = "升级%d个节点" % (i + 1)
        plt.plot(x_data, y_optimized_data_dict[i], color_warehouse[i], label=label_name, linewidth=0.6)
    plt.xticks([])
    plt.title("%s月链路剩余带宽标准差变化折线图" % month_index)
    plt.xlabel("时间")
    plt.ylabel("剩余带宽标准差")
    plt.legend(fontsize=10)
    plt.savefig("./charts/compare/variance/compare_%s_upgraded/%s.png" % (optimize_result_count, month_index))
    plt.show()


def plot_optimized_split_utilization(month_index, node_upgraded, threshold=0.0):
    json_file_origin = "utilization/add_weight/abilene_TM_2004_%s.json" % month_index
    f_0 = open(json_file_origin, 'r', encoding="utf-8")
    origin_result_dict = json.load(f_0, object_hook=dict)
    f_0.close()
    suffix = '' if node_upgraded < 2 else 's'
    json_file_optimized_nodes = "utilization/upgrade_strategy_%d_node%s/abilene_TM_2004_%s.json" \
                                % (node_upgraded, suffix, month_index)
    f_1 = open(json_file_optimized_nodes, 'r', encoding='utf-8')
    optimized_node_result_dict = json.load(f_1, object_hook=dict)
    f_1.close()
    json_file_optimized_split = "utilization/up_%d_node%s_ratio_verify/abilene_TM_2004_%s.json" \
                                % (node_upgraded, suffix, month_index)
    f_2 = open(json_file_optimized_split, 'r', encoding='utf-8')
    optimized_split_result_dict = json.load(f_2, object_hook=dict)
    f_2.close()
    x_data = []
    y0_data = []
    y1_data = []
    y2_data = []
    for key in origin_result_dict.keys():
        if origin_result_dict[key] <= threshold:
            continue
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y0_data.append(origin_result_dict[key])
        y1_data.append(optimized_node_result_dict[key])
        y2_data.append(optimized_split_result_dict[key])
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x_data, y0_data, 'r', label='原始利用率', linewidth=0.6)
    plt.plot(x_data, y1_data, 'b', label='升级%d节点后' % node_upgraded, linewidth=0.8)
    plt.plot(x_data, y2_data, 'g', label='优化分流后', linewidth=0.8)
    plt.xticks([])
    plt.title("%s月链路利用率变化折线图" % month_index)
    plt.xlabel("时间")
    plt.ylabel("链路最大利用率")
    plt.ylim((0, 1))
    plt.legend(fontsize=10)
    plt.savefig("./charts/compare/utilization/compare_%s_upgrade_split_optimized/%s.png" % (node_upgraded, month_index))
    plt.show()


def plot_optimized_split_utilization_func_val(month_index, node_upgraded, threshold=0.0):
    json_file_origin = "utilization_function_value/add_weight/abilene_TM_2004_%s.json" % month_index
    f_0 = open(json_file_origin, 'r', encoding="utf-8")
    origin_result_dict = json.load(f_0, object_hook=dict)
    f_0.close()
    suffix = '' if node_upgraded < 2 else 's'
    json_file_optimized_nodes = "utilization_function_value/upgrade_strategy_%d_node%s/abilene_TM_2004_%s.json" \
                                % (node_upgraded, suffix, month_index)
    f_1 = open(json_file_optimized_nodes, 'r', encoding='utf-8')
    optimized_node_result_dict = json.load(f_1, object_hook=dict)
    f_1.close()
    json_file_optimized_split = "utilization_function_value/up_%d_node%s_ratio_verify/abilene_TM_2004_%s.json" \
                                % (node_upgraded, suffix, month_index)
    f_2 = open(json_file_optimized_split, 'r', encoding='utf-8')
    optimized_split_result_dict = json.load(f_2, object_hook=dict)
    f_2.close()
    x_data = []
    y0_data = []
    y1_data = []
    y2_data = []
    for key in origin_result_dict.keys():
        if origin_result_dict[key] <= threshold:
            continue
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y0_data.append(origin_result_dict[key])
        y1_data.append(optimized_node_result_dict[key])
        y2_data.append(optimized_split_result_dict[key])
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.plot(x_data, y0_data, 'r', label='原始利用率', linewidth=0.6)
    plt.plot(x_data, y1_data, 'b', label='升级%d节点后' % node_upgraded, linewidth=0.8)
    plt.plot(x_data, y2_data, 'g', label='优化分流后', linewidth=0.8)
    plt.xticks([])
    plt.title("%s月链路利用率函数变化折线图" % month_index)
    plt.xlabel("时间")
    plt.ylabel("链路利用率函数")
    # plt.ylim((0, 1))
    plt.legend(fontsize=10)
    plt.savefig("./charts/compare/utilization_func_val/compare_%s_upgrade_split_optimized/%s.png" % (node_upgraded, month_index))
    plt.show()


def plot_optimized_split_variance(month_index, node_upgraded, threshold=0.0):
    json_file_origin = "variance/add_weight/abilene_TM_2004_%s.json" % month_index
    f_0 = open(json_file_origin, 'r', encoding="utf-8")
    origin_result_dict = json.load(f_0, object_hook=dict)
    f_0.close()
    suffix = '' if node_upgraded < 2 else 's'
    json_file_optimized_nodes = "variance/upgrade_strategy_%d_node%s/abilene_TM_2004_%s.json" \
                                % (node_upgraded, suffix, month_index)
    f_1 = open(json_file_optimized_nodes, 'r', encoding='utf-8')
    optimized_node_result_dict = json.load(f_1, object_hook=dict)
    f_1.close()
    json_file_optimized_split = "variance/up_%d_node%s_ratio_verify/abilene_TM_2004_%s.json" \
                                % (node_upgraded, suffix, month_index)
    f_2 = open(json_file_optimized_split, 'r', encoding='utf-8')
    optimized_split_result_dict = json.load(f_2, object_hook=dict)
    f_2.close()
    x_data = []
    y0_data = []
    y1_data = []
    y2_data = []
    for key in origin_result_dict.keys():
        if origin_result_dict[key] <= threshold:
            continue
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y0_data.append(origin_result_dict[key])
        y1_data.append(optimized_node_result_dict[key])
        y2_data.append(optimized_split_result_dict[key])
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.plot(x_data, y0_data, 'r', label='原始利用率', linewidth=0.6)
    plt.plot(x_data, y1_data, 'b', label='升级%d节点后' % node_upgraded, linewidth=0.8)
    plt.plot(x_data, y2_data, 'g', label='优化分流后', linewidth=0.8)
    plt.xticks([])
    plt.title("%s月链路剩余带宽方差变化折线图" % month_index)
    plt.xlabel("时间")
    plt.ylabel("剩余带宽方差")
    # plt.ylim((0, 1))
    plt.legend(fontsize=10)
    plt.savefig("./charts/compare/variance/compare_%s_upgrade_split_optimized/%s.png" % (node_upgraded, month_index))
    plt.show()


if __name__ == "__main__":
    # plot_utilization_compared_result('05', 2, 0.15)
    # plot_utilization_func_val_compared_result('05', 2, 0.15)
    # plot_variance_compared_result('05', 2, 0.15)
    plot_optimized_split_utilization('05', 2, 0.15)
    plot_optimized_split_utilization_func_val('05', 2, 0.15)
    plot_optimized_split_variance('05', 2, 0.15)




