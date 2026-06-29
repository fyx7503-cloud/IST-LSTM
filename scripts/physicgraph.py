# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle


def dfs(graph, start_node):
    visited = set()
    stack = [(start_node, 0)]
    result = []
    depth_dict = {'elements': [], 'depths': []}

    while stack:
        node, depth = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            depth_dict['elements'].append(node)
            depth_dict['depths'].append(depth)
            # 将邻居节点按字母顺序加入栈中
            neighbors = sorted(graph.neighbors(node), reverse=True)
            stack.extend((neighbor, depth + 1) for neighbor in neighbors)

    return result, depth_dict


# 站点列表
stations = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
    "13", "14", "15", "16", "41", "42", "43", "44", "45", "46", "47",
    "48", "49", "50", "51", "52", "53", "54", "55", "17", "18", "19",
    "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "89",
    "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100",
    "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "120",
    "121", "122", "123", "124", "125", "126", "127", "128", "56", "57",
    "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68",
    "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "80", "81", "82", "83", "84", "85", "86", "87", "88"
]

station_positions = {station: index for index, station in enumerate(stations)}

# 创建图
G = nx.Graph()

# 添加节点
for station in stations:
    G.add_node(station)

# 定义站点之间的连接（边）和颜色
connections = [
    ##一号线
    ("1", "2", "red"), ("2", "3", "red"), ("3", "4", "red"), ("4", "5", "red"), ("5", "6", "red"),
    ("6", "7", "red"), ("7", "8", "red"), ("8", "9", "red"), ("9", "10", "red"), ("10", "11", "red"),
    ("11", "12", "red"), ("12", "13", "red"), ("13", "14", "red"), ("14", "15", "red"), ("15", "16", "red"),
    ("5", "41", "red"), ("41", "42", "red"), ("42", "43", "red"), ("43", "44", "red"), ("44", "45", "red"),
    ("45", "46", "red"), ("46", "47", "red"), ("47", "48", "red"), ("48", "49", "red"), ("49", "50", "red"),
    ("50", "51", "red"), ("51", "52", "red"), ("52", "53", "red"), ("53", "54", "red"), ("54", "55", "red"),

    ##二号线
    ("17", "18", "blue"), ("18", "19", "blue"), ("19", "20", "blue"), ("20", "21", "blue"), ("21", "22", "blue"),
    ("22", "23", "blue"), ("23", "24", "blue"), ("24", "25", "blue"), ("25", "9", "blue"), ("9", "26", "blue"),
    ("26", "27", "blue"), ("27", "28", "blue"), ("28", "29", "blue"), ("29", "30", "blue"), ("30", "31", "blue"),
    ("31", "32", "blue"), ("32", "33", "blue"), ("33", "34", "blue"), ("34", "35", "blue"), ("35", "36", "blue"),
    ("36", "37", "blue"), ("37", "38", "blue"), ("38", "39", "blue"), ("39", "40", "blue"),

    ##三号线
    ("89", "90", "green"), ("90", "91", "green"), ("91", "92", "green"), ("92", "93", "green"), ("93", "94", "green"),
    ("94", "95", "green"), ("95", "96", "green"), ("96", "14", "green"), ("14", "97", "green"), ("97", "98", "green"),
    ("98", "99", "green"), ("99", "26", "green"), ("26", "100", "green"), ("100", "101", "green"),
    ("101", "102", "green"),
    ("102", "103", "green"), ("103", "104", "green"), ("104", "105", "green"), ("105", "106", "green"),
    ("106", "44", "green"),
    ("44", "107", "green"), ("107", "108", "green"), ("108", "109", "green"), ("109", "110", "green"),
    ("110", "111", "green"),
    ("111", "112", "green"), ("112", "113", "green"),

    ##四号线
    ("114", "115", "purple"), ("115", "116", "purple"), ("116", "11", "purple"), ("11", "98", "purple"),
    ("98", "117", "purple"),
    ("117", "118", "purple"), ("118", "119", "purple"), ("119", "120", "purple"), ("120", "121", "purple"),
    ("121", "122", "purple"),
    ("122", "35", "purple"), ("35", "123", "purple"), ("123", "124", "purple"), ("124", "125", "purple"),
    ("125", "126", "purple"),
    ("126", "127", "purple"), ("127", "128", "purple"),

    ##十号线
    ("1", "56", "orange"), ("56", "57", "orange"), ("57", "58", "orange"), ("58", "59", "orange"),
    ("59", "60", "orange"),
    ("60", "61", "orange"), ("61", "62", "orange"), ("62", "63", "orange"), ("63", "64", "orange"),

    ##S1
    ("65", "66", "brown"), ("66", "67", "brown"), ("67", "68", "brown"), ("68", "69", "brown"), ("69", "70", "brown"),
    ("70", "71", "brown"), ("71", "44", "brown"),

    ##S8
    ("72", "73", "pink"), ("73", "74", "pink"), ("74", "75", "pink"), ("75", "76", "pink"), ("76", "77", "pink"),
    ("77", "78", "pink"), ("78", "79", "pink"), ("79", "80", "pink"), ("80", "81", "pink"), ("81", "82", "pink"),
    ("82", "83", "pink"), ("83", "84", "pink"), ("84", "85", "pink"), ("85", "86", "pink"), ("86", "87", "pink"),
    ("87", "88", "pink"),
]

# 添加边
for connection in connections:
    G.add_edge(connection[0], connection[1], color=connection[2])

matrix_size = len(stations)
connection_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

# 填充矩阵
station_index = {station: idx for idx, station in enumerate(stations)}
for connection in connections:
    i, j = station_index[connection[0]], station_index[connection[1]]
    connection_matrix[i, j] = 1
    connection_matrix[j, i] = 1  # 因为是无向图

np.set_printoptions(threshold=np.inf)
# print(np.array_str(connection_matrix, max_line_width=np.inf))

row_index = 0
# for start_node in stations:

lineindex = []
depth = []
for start_node in stations:
    dfs_result, depth_dict = dfs(G, start_node)
    lineindex.clear()
    depth.clear()
    for element, depth_value in zip(depth_dict['elements'], depth_dict['depths']):
        if element in station_positions:
            lineindex.append(station_positions[element])
            depth.append(depth_value)
            for col_idx, dep in zip(lineindex, depth):
                connection_matrix[row_index, col_idx] = dep
    row_index += 1

shape = connection_matrix.shape
identity_matrix = np.zeros(shape)
np.fill_diagonal(identity_matrix, 1)
# ##这边先计算行和，再在对角线赋值1，然后才归一化
# row_sums = connection_matrix.sum(axis=1, keepdims=True)
connection_matrix = connection_matrix + identity_matrix
# normalized_matrix = connection_matrix / row_sums
# connection_matrix = normalized_matrix
###最后做了行归一化

with open('nanjing/graph_nj_conn.pkl', 'wb') as file:
    pickle.dump(connection_matrix, file)

print(np.array_str(connection_matrix, max_line_width=np.inf))

# # 获取边的颜色
# edges = G.edges()
# colors = [G[u][v]['color'] for u, v in edges]
#
#
# plt.figure(figsize=(12, 12))
# nx.draw(G, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold",
#         edge_color=colors)
# plt.title("地铁网络")
# plt.show()
