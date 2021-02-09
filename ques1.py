import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

influence_data=pd.read_csv('influence_data.csv')
influence_data.head()
data=influence_data[['influencer_name','follower_name']]
# data['weight']=0.1
print(data.head())

sorted(Counter(data['influencer_name']).items(), key=lambda x: x[1], reverse=True)

num = 50000
node1 = data['influencer_name'].values.tolist()[:num]
node2 = data['follower_name'].values.tolist()[:num]
print(node1)
print(node2)
G = nx.DiGraph()
print(len(node1))
for i in range(len(node1)):
    G.add_edges_from([(node1[i], node2[i])], weight=1)

x = np.zeros(len(node1))
dic = dict(zip(node1, x))

degree = G.degree
print(degree)
for i in degree:
    if (i[0] in node1):
        dic[i[0]] = i[1]
print(dic)
de = []
color_weight = {}
for i in node1:
    if(int(dic[i] >=6)):
        color_weight[i] = 5 + random.random()
    elif(int(dic[i] < 3)):
        color_weight[i] = int(dic[i]) + random.random()*3
    else:
        color_weight[i] = int(dic[i])
    de.append(int(dic[i]) * 0.05)
    # if(int(dic[i] < 3)):
    #     color_weight[i] = random.random() * 0.8
    #     de.append(random.random()* 0.8)
    # else:
    #     color_weight[i] = random.random()
    #     de.append(random.random())
print("de:" , de)
print("color:", color_weight)
# nx.draw(G, with_labels=True, node_size=8, font_size=4)
# plt.savefig('graph1.png', dpi=300)


node1 = data['influencer_name'].values.tolist()[:num]
node2 = data['follower_name'].values.tolist()[:num]
nodeweight = de
G = nx.Graph()

print(len(node1))
print(len(nodeweight))
for i in range(len(node1)):
    G.add_edges_from([(node1[i], node2[i])], weight=nodeweight[i])

# cmap = plt.cm.get_cmap('Greens')
values = [color_weight.get(node) for node in G.nodes()]
print(len(values))
print(values)

pos=nx.spring_layout(G)
# nx.draw_networkx_edge_labels(G,pos)
nx.draw(G,pos, node_color = values, with_labels=False, node_size=4,cmap=plt.cm.Reds)


# nx.draw(G, with_labels=True, node_size=8, font_size=4, node_color=values)
plt.savefig('graph_all.png', dpi=300)














# import random
#
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import Counter
# import numpy as np
#
# influence_data=pd.read_csv('subnet.csv')
# influence_data.head()
#
# data=influence_data[['influencer_name','follower_name']]
# # data['weight']=0.1
# print(data.head())
#
#
#
# sorted(Counter(data['influencer_name']).items(), key=lambda x: x[1], reverse=True)
#
#
#
# num = 11
# node1 = data['influencer_name'].values.tolist()[:num]
# node2 = data['follower_name'].values.tolist()[:num]
# print(node1)
# print(node2)
# G = nx.DiGraph()
# print(len(node1))
# for i in range(len(node1)):
#     G.add_edges_from([(node1[i], node2[i])], weight=1)
#
# x = np.zeros(len(node1))
# dic = dict(zip(node1, x))
#
# degree = G.degree
# print(degree)
# for i in degree:
#     if (i[0] in node1):
#         dic[i[0]] = i[1]
# print(dic)
# de = []
# color_weight = {}
# for i in node1:
#     for i in node1:
#         if (int(dic[i] >= 6)):
#             color_weight[i] = 5 + random.random()
#         elif (int(dic[i] < 3)):
#             color_weight[i] = int(dic[i]) + random.random() * 3
#         else:
#             color_weight[i] = int(dic[i])
#         de.append(int(dic[i]) * 0.05)
#         # if(int(dic[i] < 3)):
#         #     color_weight[i] = random.random() * 0.8
#         #     de.append(random.random()* 0.8)
#         # else:
#         #     color_weight[i] = random.random()
#         #     de.append(random.random())
#     print("de:", de)
#     print("color:", color_weight)
# # nx.draw(G, with_labels=True, node_size=8, font_size=4)
# # plt.savefig('graph1.png', dpi=300)
#
#
# node1 = data['influencer_name'].values.tolist()[:num]
# node2 = data['follower_name'].values.tolist()[:num]
# nodeweight = de
# G = nx.DiGraph()
#
# print(len(node1))
# print(len(nodeweight))
# for i in range(len(node1)):
#     G.add_edges_from([(node1[i], node2[i])], weight=nodeweight[i])
#
# # cmap = plt.cm.get_cmap('Greens')
# values = [color_weight.get(node) for node in G.nodes()]
# print(len(values))
#
# pos=nx.spring_layout(G)
# # nx.draw_networkx_edge_labels(G,pos)
# nx.draw(G,pos, node_color = values, with_labels=True, node_size=200, font_size=12,cmap=plt.cm.Reds)
#
#
# # nx.draw(G, with_labels=True, node_size=8, font_size=4, node_color=values)
# plt.savefig('graph1sub.png', dpi=300)