import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist

genres = pd.read_csv('influence_data.csv')
key = []
value = []
for i in genres["influencer_id"]:
    key.append(i)
for j in genres["influencer_main_genre"]:
    value.append(j)
for i in genres["follower_id"]:
    key.append(i)
for j in genres["follower_main_genre"]:
    value.append(j)
dic = dict(zip(key,value))   #id和genres一一对应
# print(dic)
newdic = []
for i in dic:
    if(dic[i] == "Jazz"):
        newdic.append(i)
print(newdic)



new_music = []
new_music_popu = []
data_music = pd.read_csv('full_music_data.csv')
data_music = data_music.values.tolist()
for i in data_music:
    temp = i[1][1:-1]
    if (len(temp) <= 7):
        if (int(temp) in newdic):
            new_music.append(i[2:-3])
    else:
        temp = temp.split(",")
        if (int(temp[0]) in newdic):
            new_music.append(i[2:-3])
        if (int(temp[1]) in newdic):
            new_music.append(i[2:-3])

    if(i[-3] >1957 and i[-3]<1963):
        temp = i[1][1:-1]
        if (len(temp) <= 7):
            if (int(temp) in newdic):
                new_music_popu.append(i[2:-3])
        else:
            temp = temp.split(",")
            if(int(temp[0]) in newdic):
                new_music_popu.append(i[2:-3])
            if(int(temp[1]) in newdic):
                new_music_popu.append(i[2:-3])
new_music = np.array(new_music)
aver = new_music.mean(axis=0)
print(new_music.max(axis=0) - new_music.min(axis=0))
aver = aver / (new_music.max(axis=0) - new_music.min(axis=0))
print(aver)


new_music_popu = np.array(new_music_popu)
popu_aver = new_music_popu.mean(axis=0)
chazhi = new_music_popu.max(axis=0) - new_music_popu.min(axis=0)
for i in range(len(chazhi)):
    if chazhi[i]==0:
        chazhi[i] = 1
popu_aver = popu_aver / chazhi
print(popu_aver)


cha = popu_aver - aver
print(cha)

danceability = 0.06911291
tempo = 0.10886683
duration_ms = 0.1277921
s = float(danceability) + float(tempo) + float(duration_ms)
print(s)
danceability_rate = danceability / s
tempo_rate = tempo / s
duration_ms_rate = duration_ms / s

print(danceability_rate, tempo_rate , duration_ms_rate)







danceability_rate = 0.226
tempo_rate = 0.356
duration_ms_rate = 0.418

data_music = pd.read_csv('influence_data.csv')
data_music = data_music.values.tolist()
name = []
for i in data_music[:]:
    if(i[2] == "Jazz" and (i[3] <= 1970)):
        name.append(i[0])
print(len(name))


id = []
danceability = []
tempo = []
duration_ms = []

data_music = pd.read_csv('data_by_artist.csv')
data_music = data_music.values.tolist()
for i in data_music:
    if (i[1] in name):
        id.append(i[1])
        danceability.append(i[2])
        tempo.append(i[5])
        duration_ms.append(i[-3])
print(id)
print(len(id))
danceability = danceability / chazhi[0] - popu_aver[0]
tempo = tempo / chazhi[3] - popu_aver[3]
duration_ms = duration_ms / chazhi[-2] - popu_aver[-2]
allinall = abs(danceability) * danceability_rate + abs(tempo) * tempo_rate + abs(duration_ms) * duration_ms_rate
allinall = list(allinall)
print(min(allinall))
print(allinall.index(min(allinall)))
index = allinall.index(min(allinall))
print(id[index])

del id[index]
del allinall[index]
print(min(allinall))
print(allinall.index(min(allinall)))
index = allinall.index(min(allinall))
print(id[index])

print(id.index(928942))
print(allinall[id.index(928942)])

# print(dic)
#
# danceability = []
# tempo = []
# duration_ms = []
#
# for i in data_music[:]:
#     if (i[0] in dic.keys()):
#         dic[i[0]] += 1
# id = list(dic.keys())
# number = list(dic.values())
# print(id)
# print(number)
# print(len(id))
# niandai = []
# for i in dic.keys():
#     for j in data_music[:]:
#         if(i == j[0]):
#             niandai.append(j[3])
#             break
# print(niandai)
# print(len(niandai))



