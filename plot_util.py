import matplotlib.pyplot as plt
import numpy as np
import json

color_warehouse = ['blue', 'green', 'blueviolet']


def plot_training_result(month_index):
    json_file = "ecmp_utilization/add_weight/abilene_TM_2004_%s.json" % month_index
    f = open(json_file, 'r', encoding="utf-8")
    result_dict = json.load(f, object_hook=dict)
    f.close()
    x_data = []
    y_data = []
    for key in result_dict.keys():
        day_index = key.split('-')[-2:]
        # if day_index[1] == "0000.xml":
        #     x_data.append(day_index[0])
        # else:
        #     x_data.append("")
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y_data.append(result_dict[key])
    plt.figure(figsize=(20, 5))
    plt.plot(x_data, y_data, linewidth=0.8)
    plt.xticks([])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("%s月链路利用率变化折线图" % month_index)
    plt.xlabel("date")
    plt.ylabel("max_utilization")
    plt.ylim((0, 1))
    plt.savefig("./charts/origin/%s.png" % month_index)


def plot_utilization_compared_result(month_index, optimize_result_count):
    json_file_origin = "ecmp_utilization/add_weight/abilene_TM_2004_%s.json" % month_index
    optimized_results = []
    for i in range(optimize_result_count):
        json_file_optimized = "ecmp_utilization/upgrade_strategy_%d_nodes/abilene_TM_2004_%s.json" % ((2 + i), month_index)
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
        x_data.append(''.join(key.split('-')[-2:]).split('.')[0])
        y1_data.append(origin_result_dict[key])
        for i in range(optimize_result_count):
            if i in y_optimized_data_dict.keys():
                y_optimized_data_dict[i].append(optimized_results[i][key])
            else:
                y_optimized_data_dict[i] = [optimized_results[i][key]]
    plt.figure(figsize=(20, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x_data, y1_data, 'r', label='原始利用率', linewidth=0.8)
    for i in range(optimize_result_count):
        label_name = "升级%d个节点" % (i + 2)
        plt.plot(x_data, y_optimized_data_dict[i], color_warehouse[i], label=label_name, linewidth=0.8)
    plt.xticks([])
    plt.title("%s月链路利用率变化折线图" % month_index)
    plt.xlabel("date")
    plt.ylabel("max_utilization")
    plt.ylim((0, 1))
    plt.legend(fontsize=10)
    # plt.savefig("./charts/compare_3_upgraded/%s.png" % month_index)
    plt.show()


if __name__ == "__main__":
    plot_utilization_compared_result("09", 3)
    months = ["03", "04", "05", "06", "07", "08", "09"]
    # for month in months:
    #     plot_training_result(month)
