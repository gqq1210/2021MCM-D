import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist

#---------------------------------排序跟随者人数---------------------------------

# data_music = pd.read_csv('influence_data.csv')
# data_music = data_music.values.tolist()
#
# key = []
# value = []
# dict = {}
# for i in data_music[:]:
#     if(i[2] == "Jazz"):
#         if(i[0] in key):
#             dict[i[0]] += 1
#         else:
#             dict[i[0]] = 1
#             key.append(i[0])
# kk = sorted(dict.items(), key=lambda item:item[1])
# print(kk)






# data_music = pd.read_csv('full_music_data.csv')
# data_music = data_music.values.tolist()
# genres = pd.read_csv('influence_data.csv')
# genres = genres.values.tolist()
#
# key = []
# for i in genres:
#     if (i[2] == "Jazz"):
#         key.append(i[0])
#     if (i[-2] == "Jazz"):
#         key.append(i[-4])
#
# print(key)
# tempo = []
# danceability = []
# dura = []
# year = []
#
# for i in data_music:
#     name = i[1][1:-1]
#     if(len(name) <= 7):
#         if(int(name) in key):
#             danceability.append(i[2])
#             tempo.append(i[5])
#             dura.append(i[-5])
#             year.append(i[-3])

data_music = pd.read_csv('ques6_bianhua.csv')
danceability = data_music['danceability'].values.tolist()
tempo = data_music['tempo'].values.tolist()
year = data_music['year'].values.tolist()
dura = data_music['duration_ms'].values.tolist()





fig = plt.figure(figsize=(30,8))
ax1 = fig.add_subplot(131)

plt.xlabel('year', size=25)
plt.ylabel('danceability', size=25)
y = danceability
x = year

colors1 = '#00CED1'  # 点的颜色
colors2 = '#DC143C'
plt.tick_params(labelsize=17)
plt.scatter(x, y, color=colors1)
# plt.legend()
# plt.savefig('ques6_danceability.png', dpi=300)
# plt.show()


# fig = plt.figure()
ax1 = fig.add_subplot(132)
plt.xlabel('year', size=25)
plt.ylabel('tempo', size=25)
y = tempo
x = year

colors1 = '#00CED1'  # 点的颜色
colors2 = '#DC143C'
plt.tick_params(labelsize=17)
plt.scatter(x, y, color=colors1)
# plt.legend()
# plt.savefig('ques6_tempo.png', dpi=300)
# plt.show()



# fig = plt.figure()
ax1 = fig.add_subplot(133)
plt.xlabel('year', size=25)
plt.ylabel('duration_ms', size=25)
y = dura
x = year

colors1 = '#00CED1'  # 点的颜色
colors2 = '#DC143C'
plt.scatter(x, y, color=colors1)
# plt.legend()
plt.tick_params(labelsize=17)
plt.savefig('ques6_all_artist.png', dpi=300)
plt.show()