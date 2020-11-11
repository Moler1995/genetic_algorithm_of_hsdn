import xml.etree.ElementTree as ET
import numpy as np

city_to_num_dict = {"ATLA-M5": 2, "ATLAng": 3, "CHINng": 11, "DNVRng": 8, "HSTNng": 4, "IPLSng": 10,
                    "KSCYng": 9, "LOSAng": 5, "NYCMng": 0, "SNVAng": 6, "STTLng": 7, "WASHng": 1}
num_to_city_dict = {2: "ATLA-M5", 3: "ATLAng", 11: "CHINng", 8: "DNVRng", 4: "HSTNng", 10: "IPLSng",
                    9: "KSCYng", 5: "LOSAng", 0: "NYCMng", 6: "SNVAng", 7: "STTLng", 1: "WASHng"}
max_val = float('inf')


def parse_traffics(filename):
    elements = ET.parse(filename)
    count = len(elements.getroot()[0])
    traffic = np.zeros([count, count])
    for src_city in elements.getroot()[0]:
        src_city_index = city_to_num_dict[src_city.attrib["id"]]
        for tg_city in src_city:
            if tg_city.attrib["id"] == src_city.attrib["id"]:
                continue
            tg_city_index = city_to_num_dict[tg_city.attrib["id"]]
            traffic[src_city_index][tg_city_index] = tg_city.text
    return traffic


def parse_weights(graph):
    graph_xml = ET.parse("abilene/Abilene-Topo-10-04-2004.xml")
    link_nodes = graph_xml.getroot()[2][0]
    weight_dict = {}
    for link_node in link_nodes:
        link_name = link_node.attrib['id']
        split_link = str(link_name).split(',')
        if city_to_num_dict[split_link[0]] <= city_to_num_dict[split_link[1]]:
            continue
        link_weight = int(link_node[0][0].text)
        weight_dict[link_name] = link_weight
    weight_matrix = []
    node_count = len(graph)
    for i in range(node_count):
        for j in range(node_count):
            if i <= j or graph[i][j] == max_val:
                continue
            link_name = ','.join([num_to_city_dict[i], num_to_city_dict[j]])
            weight_matrix.append(weight_dict[link_name])
    return weight_matrix


if __name__ == "__main__":
    # print(parse_traffics("abilene/TM-2004-09-10-2030.xml"))
    graph = np.ones([12, 12]) * max_val
    graph[range(len(graph)), range(len(graph))] = 0
    graph[0][1], graph[0][11], graph[1][0], graph[1][3], graph[2][3], graph[3][1], graph[3][2], graph[3][4], \
        graph[3][10], graph[4][3], graph[4][5], graph[4][9], graph[5][4], graph[5][6], graph[6][5], \
        graph[6][7], graph[6][8], graph[7][6], graph[7][8], graph[8][7], graph[8][6], graph[8][9], \
        graph[9][8], graph[9][4], graph[9][10], graph[10][9], graph[10][3], graph[10][11], graph[11][10], graph[11][0] \
        = [1] * 30
    print(parse_weights(graph))
