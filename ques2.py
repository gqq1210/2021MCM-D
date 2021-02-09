import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist

data_music = pd.read_csv('data_by_artist.csv')
data_music = data_music.values.tolist()
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
print("音乐家数量", len(dic))


count = 0
print(len(data_music))
for i in range(len(data_music)):
    data_music[i].append(dic.get(data_music[i][1]))
    if dic.get(data_music[i][1])==None:
        print(data_music[i][1])
        count = count + 1
print(count)


for i in data_music[:]:
    if(i[-1] == None):
        data_music.remove(i)
print(len(data_music))
print(data_music)

list_genres=[]
for i in data_music[:]:
    if(i[-1] not in list_genres):
        list_genres.append(i[-1])
print(len(list_genres))
print(list_genres)
print("---------------------------------")
Vocal = []
Country = []
Pop = []
Classical = []
Jazz = []
International = []
Reggae = []
RandB = []
Latin = []
Stage = []
Avant = []
Easy = []
Blues = []
Folk = []
Religious = []
Electronic = []
NewAge = []
Comedy = []
Children = []
UnKnown = []
for music in data_music:
    if(music[-1]=="Vocal"):
        Vocal.append(music[2:-2])
    elif(music[-1]=="Country"):
        Country.append(music[2:-2])
    elif (music[-1] == "Pop/Rock"):
        Pop.append(music[2:-2])
    elif (music[-1] == "Classical"):
        Classical.append(music[2:-2])
    elif (music[-1] == "Jazz"):
        Jazz.append(music[2:-2])
    elif (music[-1] == "International"):
        International.append(music[2:-2])
    elif (music[-1] == "Reggae"):
        Reggae.append(music[2:-2])
    elif (music[-1] == "R&B;"):
        RandB.append(music[2:-2])
    elif (music[-1] == "Latin"):
        Latin.append(music[2:-2])
    elif (music[-1] == "Stage & Screen"):
        Stage.append(music[2:-2])
    elif (music[-1] == "Avant-Garde"):
        Avant.append(music[2:-2])
    elif (music[-1] == "Easy Listening"):
        Easy.append(music[2:-2])
    elif (music[-1] == "Blues"):
        Blues.append(music[2:-2])
    elif (music[-1] == "Folk"):
        Folk.append(music[2:-2])
    elif (music[-1] == "Religious"):
        Religious.append(music[2:-2])
    elif (music[-1] == "Electronic"):
        Electronic.append(music[2:-2])
    elif (music[-1] == "New Age"):
        NewAge.append(music[2:-2])
    elif (music[-1] == "Comedy/Spoken"):
        Comedy.append(music[2:-2])
    elif (music[-1] == "Children's"):
        Children.append(music[2:-2])
    elif (music[-1] == "Unknown"):
        UnKnown.append(music[2:-2])

# Vocal_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(i+1, len(Vocal)):
#         Vocal_pearsonr.append(pearsonr(Vocal[i], Vocal[j])[0])
# print(np.mean(Vocal_pearsonr))
# # print("Vocal_pearsonr:",np.mean(Vocal_pearsonr))
#
# # Country_pearsonr = []
# # for i in range(len(Country)):
# #     for j in range(i+1, len(Country)):
# #         Country_pearsonr.append(pearsonr(Country[i], Country[j])[0])
# # print("Country_pearsonr:",np.mean(Country_pearsonr))
#
#
# Vocal_Country_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(Country)):
#         Vocal_Country_pearsonr.append(pearsonr(Vocal[i], Country[j])[0])
# print(np.mean(Vocal_Country_pearsonr))
# # print("Vocal_Country_pearsonr:", np.mean(Vocal_Country_pearsonr))
#
# Vocal_Pop_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(Pop)):
#         Vocal_Pop_pearsonr.append(pearsonr(Vocal[i], Pop[j])[0])
# print(np.mean(Vocal_Pop_pearsonr))
# # print("Vocal_Pop_pearsonr:", np.mean(Vocal_Pop_pearsonr))
#
# Vocal_Classical_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(Classical)):
#         Vocal_Classical_pearsonr.append(pearsonr(Vocal[i], Classical[j])[0])
# print(np.mean(Vocal_Classical_pearsonr))
# # print("Vocal_Classical_pearsonr:", np.mean(Vocal_Classical_pearsonr))
#
# Vocal_Jazz_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(Jazz)):
#         Vocal_Jazz_pearsonr.append(pearsonr(Vocal[i], Jazz[j])[0])
# print(np.mean(Vocal_Jazz_pearsonr))
# # print("Vocal_Jazz_pearsonr:", np.mean(Vocal_Jazz_pearsonr))
#
#
# Vocal_Religious_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(Religious)):
#         Vocal_Religious_pearsonr.append(pearsonr(Vocal[i], Religious[j])[0])
# print(np.mean(Vocal_Religious_pearsonr))
#
#
# Vocal_Electronic_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(Electronic)):
#         Vocal_Electronic_pearsonr.append(pearsonr(Vocal[i], Electronic[j])[0])
# print(np.mean(Vocal_Electronic_pearsonr))
#
# Vocal_NewAge_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(NewAge)):
#         Vocal_NewAge_pearsonr.append(pearsonr(Vocal[i], NewAge[j])[0])
# print(np.mean(Vocal_NewAge_pearsonr))
#
# Vocal_Comedy_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(Comedy)):
#         Vocal_Comedy_pearsonr.append(pearsonr(Vocal[i], Comedy[j])[0])
# print(np.mean(Vocal_Comedy_pearsonr))
#
#
#
# Vocal_Children_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(Children)):
#         Vocal_Children_pearsonr.append(pearsonr(Vocal[i], Children[j])[0])
# print(np.mean(Vocal_Children_pearsonr))
#
#
# Vocal_Unknown_pearsonr = []
# for i in range(len(Vocal)):
#     for j in range(len(UnKnown)):
#         Vocal_Unknown_pearsonr.append(pearsonr(Vocal[i], UnKnown[j])[0])
# print(np.mean(Vocal_Unknown_pearsonr))



def cal(gen1, gen2):
    gen1_gen2 = []
    for i in range(len(gen1)):
        for j in range(len(gen2)):
            gen1_gen2.append(pearsonr(gen1[i], gen2[j])[0])
    print(np.mean(gen1_gen2))


def cal_all(Vocal):
    cal(Vocal, Vocal)
    cal(Vocal, Country)
    cal(Vocal, Pop)
    cal(Vocal, Classical)
    cal(Vocal, Jazz)
    cal(Vocal, International)
    cal(Vocal, Reggae)
    cal(Vocal, RandB)
    cal(Vocal, Latin)
    cal(Vocal, Stage)
    cal(Vocal, Avant)
    cal(Vocal, Easy)
    cal(Vocal, Blues)
    cal(Vocal, Folk)
    cal(Vocal, Religious)
    cal(Vocal, Electronic)
    cal(Vocal, NewAge)
    cal(Vocal, Comedy)
    cal(Vocal, Children)


cal_all(Folk)
print("-----")

cal_all(Religious)
print("-----")

cal_all(Electronic)
print("-----")

cal_all(NewAge)
print("-----")

cal_all(Comedy)
print("-----")

cal_all(Children)
print("-----")





def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


# print(cosine_similarity(Vocal[0], Vocal[1]))
# print(cosine_similarity(Vocal[0], Pop[10]))
# print(pdist(Vocal))
# print(min(pdist(Vocal)))
# print("Vocal余弦相似度", np.mean(pdist(Vocal)))
