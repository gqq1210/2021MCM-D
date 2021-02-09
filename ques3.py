import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist

# 计算A影响B的人数，A是影响者，B是跟随者
def cal(A, B):
    music = 0
    data_music = pd.read_csv('influence_data.csv')
    data = data_music.values.tolist()
    for i in data[:]:
        if(i[2] == A and i[6] == B):
            music += 1
    print(music)
    return music


genres = ['Vocal', 'Country', 'Pop/Rock', 'Classical', 'Jazz', 'International', 'Reggae', 'R&B;', 'Latin', 'Stage & Screen', 'Avant-Garde', 'Easy Listening', 'Blues', 'Folk', 'Religious', 'Electronic', 'New Age', 'Comedy/Spoken', "Children's"]

def cal_all(Vocal):
    all = []
    for i in range(len(genres)):
        all.append(cal(Vocal, genres[i]))
    print("--------")
    return all


fig = plt.figure()
x = genres
for i in genres:
    y = cal_all(i)
    plt.plot(x, y, label=i)
plt.tight_layout()
# plt.yscale("log")
plt.xticks(rotation=70) # 倾斜70度
plt.xlabel("genres")
plt.ylabel("numbers")
plt.legend(loc='upper right', fontsize=5)
plt.savefig('ques3_genres_number.png', bbox_inches="tight", dpi=300)
plt.show()








## 大的折线图
# data_music = pd.read_csv('full_music_data.csv')
# data_music = data_music.values.tolist()
# genres = pd.read_csv('influence_data.csv')
# key = []
# value = []
# for i in genres["influencer_id"]:
#     key.append(i)
# for j in genres["influencer_main_genre"]:
#     value.append(j)
# for i in genres["follower_id"]:
#     key.append(i)
# for j in genres["follower_main_genre"]:
#     value.append(j)
# dic = dict(zip(key,value))   #id和genres一一对应
# # print(dic)
#
#
# music = []
# for i in range(len(data_music)):
#     if(len(data_music[i][1][1:-1]) <= 7):
#         music.append([dic.get(int(data_music[i][1][1:-1])), data_music[i][-3]])
# print(len(music))
# # print(music)
#
#
# for i in music[:]:
#     if(i[0] == None):
#         music.remove(i)
# print(len(music))
#
# music_dic = {}
# for i in music[:]:
#     if(i[0] not in music_dic):
#         music_dic[i[0]] = []
#     else:
#         music_dic[i[0]].append(i)
#
# def popularity_time(name, color):
#     pop_time = {}
#     for i in music_dic[name]:
#         if (i[1] not in pop_time):
#             pop_time[i[1]] = 1
#         else:
#             pop_time[i[1]] += 1
#     new_time = sorted(pop_time.items())
#     x = []
#     for i in new_time[:]:
#         x.append(i[0])
#     y = []
#     for i in new_time[:]:
#         y.append(i[1])
#     plt.plot(x, y, color = color, label = name)
#
#
# fig = plt.figure()

# popularity_time("Vocal", 'red')
# popularity_time("Country", 'yellow')
# popularity_time("Pop/Rock", 'blue')
# popularity_time("Classical", 'black')
# popularity_time("Jazz", 'pink')
# popularity_time("International", 'green')
#
# popularity_time("Reggae", 'darkorange')
# popularity_time("R&B;", 'tan')
# popularity_time("Latin", 'darkgoldenrod')
# popularity_time("Stage & Screen", 'brown')
# popularity_time("Avant-Garde", 'darkred')
# popularity_time("Easy Listening", 'slategrey')
#
# popularity_time("Blues", 'dimgray')
# popularity_time("Folk", 'lime')
# popularity_time("Religious", 'blueviolet')
# popularity_time("Electronic", 'plum')
# popularity_time("New Age", 'c')
# popularity_time("Comedy/Spoken", 'teal')

# popularity_time("Children's", 'purple')
# popularity_time("Unknown", 'olivedrab')

# plt.title("R&B")
# plt.tight_layout()
# plt.xlabel("year")
# plt.ylabel("number of songs")
# plt.legend(loc='upper right', fontsize=5)
# plt.savefig('ques3_popularity_time.png', bbox_inches="tight", dpi=300)
# plt.show()