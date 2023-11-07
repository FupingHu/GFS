import pickle
import numpy as np

feature_matrices = open('feature_matrices.pkl', 'rb')
feature_matrices = pickle.load(feature_matrices)
# print(len(feature_matrices))

adjacency_matrices = open('adjacency_matrices.pkl', 'rb')
adjacency_matrices = pickle.load(adjacency_matrices)
# print(len(adjacency_matrices))
# print(adjacency_matrices)

edge_feature_matrices = open('edge_feature_matrices.pkl', 'rb')
edge_feature_matrices = pickle.load(edge_feature_matrices)
# print(len(edge_feature_matrices))
# print(adjacency_matrices[0])

labels = np.load('labels.npy', encoding="latin1")
# print(len(labels))

if (len(feature_matrices) == len(adjacency_matrices)) & (len(adjacency_matrices) == len(edge_feature_matrices)) & (
        len(edge_feature_matrices) == len(labels)):
    print('Dataset Checked! Pass')

Num_Graph = len(labels)
dataset = 'esol'
DS_A = dataset + '_A.txt'
DS_graph_attributes = dataset + '_graph_attributes.txt'
DS_graph_indicator = dataset + '_graph_indicator.txt'
DS_node_attributes = dataset + '_node_attributes.txt'

with open(DS_graph_attributes, 'w') as f:
    for num in range(Num_Graph):
        f.writelines(str(labels[num]).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n')
#
with open(DS_graph_indicator, 'w') as f:
    for num in range(Num_Graph):
        graph_indicator = num + 1
        for node_index in range(len(feature_matrices[num])):
            f.writelines(str(graph_indicator) + '\n')

with open(DS_node_attributes, 'w') as f:
    for num in range(Num_Graph):
        for node_index in range(len(feature_matrices[num])):
            # print(feature_matrices[num][node_index])
            node_fea_str = ''
            for node_fea_index in range(len(feature_matrices[num][node_index])):
                if node_fea_index == 0:
                    node_fea_str += str(feature_matrices[num][node_index][node_fea_index])
                else:
                    node_fea_str += ', '
                    node_fea_str += str(feature_matrices[num][node_index][node_fea_index])
            print(node_fea_str)
            f.writelines(node_fea_str + '\n')

# print(adjacency_matrices[0])
with open(DS_A, 'w') as f:
    node_index_total = 0
    for num in range(Num_Graph):
        # NonZero_Num = num + 1
        for adj_index, adj in enumerate(adjacency_matrices[num]):
            node_index_total += 1
            node_index = adj_index + 1
            # NonZero_Index += NonZero_Num

            print(node_index_total)
            print(node_index)
            A_dict = dict(adj.todok())
            for key, value in enumerate(A_dict):
                # print(key)
                target_node_index = list(value)[1] + 1
                f.writelines(
                    '{0}, {1}'.format(node_index_total, node_index_total + (target_node_index - node_index)) + '\n')
                # if node_index == target_node_index:
                #     pass
                # else:
                #     print('{0}, {1}'.format(node_index, target_node_index))
                #     print('{0}, {1}'.format(node_index_total, node_index_total + (target_node_index - node_index)))
#
#
#         # for adj in range(adjacency_matrices[num].shape[0]):
#         #     for edge in adj:
#         #     edge = str(adjacency_matrices[num][index]).replace('\'', '')[:-1]
#         #     edge += '\n'
#         #     print(edge)
#         #     f.writelines(str(adjacency_matrices[num][index]).replace('\'', '') + '\n')

# for num in range(Num_Graph):
#     graph_indicator = num + 1

# print(graph_indicator)
# print(adjacency_matrices[num])
# print(feature_matrices[num])
# print(edge_feature_matrices[num])
# print(labels[num])
