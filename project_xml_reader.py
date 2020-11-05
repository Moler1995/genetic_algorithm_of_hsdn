import xml.etree.ElementTree as ET
import numpy as np

city_to_num_dict = {"ATLA-M5": 2, "ATLAng": 3, "CHINng": 11, "DNVRng": 8, "HSTNng": 4, "IPLSng": 10,
                    "KSCYng": 9, "LOSAng": 5, "NYCMng": 0, "SNVAng": 6, "STTLng": 7, "WASHng": 1}


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


if __name__ == "__main__":
    print(parse_traffics("abilene/TM-2004-09-10-2030.xml"))

