import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist

# data_music = pd.read_csv('influence_data.csv')
# data_music = data_music.values.tolist()
# genres = pd.read_csv('influence_data.csv')
# key = []
# value = []
# for i in genres["influencer_id"]:
#     key.append(i)
# for j in genres["influencer_main_genre"]:
#     value.append(j)
# dic = dict(zip(key,value))   #id和genres一一对应
# dic_new = sorted(dic.keys())
# print(dic)
# print(dic_new)
#
# repeat = []
# for i in dic_new:
#     for j in data_music:
#         if(j[0] == i):
#             if(j[-2] != dic.get(i)):
#                 if (i not in repeat):
#                     repeat.append(i)
# print(repeat)
# print(len(repeat))
#
# key = []
# value = []
# for i in repeat:
#     count = 0
#     notalong = 0
#     for j in data_music:
#         if (j[0] == i):
#             count += 1
#             if (j[-2] != dic.get(i)):
#                 notalong += 1
#     key.append(i)
#     value.append(notalong / count)
# dic = dict(zip(key,value))
# print(dic)
# new_dic = sorted(dic.items())
# print(new_dic)
# newdic = []
#
# kkk = []
# for i in new_dic:
#     newdic.append(i[1])
#     if(i[1] == 1):
#         kkk.append(i[0])
# print(kkk)
# print(newdic)
# print(np.mean(newdic))
# print(np.min(newdic))
# print(np.max(newdic))
#
# fig = plt.figure()
# x = np.arange(len(newdic))
# y = newdic
# plt.scatter(x, y)
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("difference rate of genres following")
# plt.legend(loc='upper right', fontsize=5)
# plt.savefig('ques4_difference_rate.png', bbox_inches="tight", dpi=300)
# plt.show()






# data_music = pd.read_csv('full_music_data.csv')
# data_music = data_music.values.tolist()
# danceability=[]
# energy=[]
# valence=[]
# tempo=[]
# loudness=[]
# mode=[]
# key=[]
# acousticness=[]
# instrumentalness=[]
# liveness=[]
# speechiness=[]
# explicit=[]
# duration_ms=[]
# popularity=[]
# for i in data_music[:]:
#     danceability.append(i[2])
#     energy.append(i[3])
#     valence.append(i[4])
#     tempo.append(i[5])
#     loudness.append(i[6])
#     mode.append(i[7])
#     key.append(i[8])
#     acousticness.append(i[9])
#     instrumentalness.append(i[10])
#     liveness.append(i[11])
#     speechiness.append(i[12])
#     explicit.append(i[13])
#     duration_ms.append(i[14])
#     popularity.append(i[15])
#
#
# all = []
# all.append(pearsonr(danceability, popularity)[0])
# all.append(pearsonr(energy, popularity)[0])
# all.append(pearsonr(valence, popularity)[0])
# all.append(pearsonr(tempo, popularity)[0])
# all.append(pearsonr(loudness, popularity)[0])
# all.append(pearsonr(mode, popularity)[0])
# all.append(pearsonr(key, popularity)[0])
# all.append(pearsonr(acousticness, popularity)[0])
# all.append(pearsonr(instrumentalness, popularity)[0])
# all.append(pearsonr(liveness, popularity)[0])
# all.append(pearsonr(speechiness, popularity)[0])
# all.append(pearsonr(explicit, popularity)[0])
# all.append(pearsonr(duration_ms, popularity)[0])
# print(all)
# for i in range(len(all)):
#     all[i] = abs(all[i])
# print(all)







# 因子分析法
import pandas as pd
import numpy as np
import math as math
import numpy as np
from numpy import *
from scipy.stats import bartlett
from factor_analyzer import *
import numpy.linalg as nlg
from sklearn.cluster import KMeans
from matplotlib import cm
import matplotlib.pyplot as plt
def main():
    df=pd.read_csv("ques4/data3.csv")
    # print(df)
    df2=df.copy()
    # print("\n原始数据:\n",df2)
    # del df2['artist_names']
    # del df2['artists_id']
    # del df2['year']
    # del df2['release_date']
    # del df2['song_title (censored)']
    # print(df2)
    # 皮尔森相关系数
    df2_corr=df2.corr()
    print("\n相关系数:\n",df2_corr)
    #热力图
    cmap = cm.Blues
    # cmap = cm.hot_r
    fig = plt.figure()
    ax=fig.add_subplot(111)
    map = ax.imshow(df2_corr, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title('correlation coefficient--headmap')
    ax.set_yticks(range(len(df2_corr.columns)))
    ax.set_yticklabels(df2_corr.columns)
    ax.set_xticks(range(len(df2_corr)))
    ax.set_xticklabels(df2_corr.columns)
    plt.colorbar(map)
    # plt.savefig('ques4_q2.png', bbox_inches="tight", dpi=300)
    plt.show()
    # KMO测度
    def kmo(dataset_corr):
        corr_inv = np.linalg.inv(dataset_corr)
        nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
        A = np.ones((nrow_inv_corr, ncol_inv_corr))
        for i in range(0, nrow_inv_corr, 1):
            for j in range(i, ncol_inv_corr, 1):
                A[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
                A[j, i] = A[i, j]
        dataset_corr = np.asarray(dataset_corr)
        kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
        kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
        kmo_value = kmo_num / kmo_denom
        return kmo_value
    print("\nKMO测度:", kmo(df2_corr))
    # 巴特利特球形检验
    df2_corr1 = df2_corr.values
    print("\n巴特利特球形检验:", bartlett(df2_corr1[0], df2_corr1[1], df2_corr1[2], df2_corr1[3], df2_corr1[4],
                                  df2_corr1[5], df2_corr1[6], df2_corr1[7], df2_corr1[8], df2_corr1[9],
                                  df2_corr1[10], df2_corr1[11]))
    # 求特征值和特征向量
    eig_value, eigvector = nlg.eig(df2_corr)  # 求矩阵R的全部特征值，构成向量
    eig = pd.DataFrame()
    eig['names'] = df2_corr.columns
    eig['eig_value'] = eig_value
    eig.sort_values('eig_value', ascending=False, inplace=True)
    print("\n特征值\n：",eig)
    eig1=pd.DataFrame(eigvector)
    eig1.columns = df2_corr.columns
    eig1.index = df2_corr.columns
    print("\n特征向量\n",eig1)
    # 求公因子个数m,使用前m个特征值的比重大于85%的标准，选出了公共因子是五个
    for m in range(1, 12):
        if eig['eig_value'][:m].sum() / eig['eig_value'].sum() >= 0.85:
            print("\n公因子个数:", m)
            break
    # # 因子载荷阵
    # A = np.mat(np.zeros((15, 5)))
    # i = 0
    # j = 0
    # while i < 5:
    #     j = 0
    #     while j < 15:
    #         A[j:, i] = sqrt(eig_value[i]) * eigvector[j, i]
    #         j = j + 1
    #     i = i + 1
    # a = pd.DataFrame(A)
    # a.columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    # a.index = df2_corr.columns
    # print("\n因子载荷阵\n", a)
    # fa = FactorAnalyzer(n_factors=5)
    # fa.loadings_ = a
    # # print(fa.loadings_)
    # print("\n特殊因子方差:\n", fa.get_communalities())  # 特殊因子方差，因子的方差贡献度 ，反映公共因子对变量的贡献
    # var = fa.get_factor_variance()  # 给出贡献率
    # print("\n解释的总方差（即贡献率）:\n", var)
    # # 因子旋转
    # rotator = Rotator()
    # b = pd.DataFrame(rotator.fit_transform(fa.loadings_))
    # b.columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    # b.index = df2_corr.columns
    # print("\n因子旋转:\n", b)
    # # 因子得分
    # X1 = np.mat(df2_corr)
    # X1 = nlg.inv(X1)
    # b = np.mat(b)
    # factor_score = np.dot(X1, b)
    # factor_score = pd.DataFrame(factor_score)
    # factor_score.columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    # factor_score.index = df2_corr.columns
    # print("\n因子得分：\n", factor_score)
    # fa_t_score = np.dot(np.mat(df2), np.mat(factor_score))
    # print("\n应试者的五个因子得分：\n",pd.DataFrame(fa_t_score))
    # # 综合得分
    # wei = [[0.50092], [0.137087], [0.097055], [0.079860], [0.049277]]
    # fa_t_score = np.dot(fa_t_score, wei) / 0.864198
    # fa_t_score = pd.DataFrame(fa_t_score)
    # fa_t_score.columns = ['综合得分']
    # fa_t_score.insert(0, 'ID', range(1, 49))
    # print("\n综合得分：\n", fa_t_score)
    # print("\n综合得分：\n", fa_t_score.sort_values(by='综合得分', ascending=False).head(6))
    # plt.figure()
    # ax1=plt.subplot(111)
    # X=fa_t_score['ID']
    # Y=fa_t_score['综合得分']
    # plt.bar(X,Y,color="#87CEFA")
    # # plt.bar(X, Y, color="red")
    # plt.title('result00')
    # ax1.set_xticks(range(len(fa_t_score)))
    # ax1.set_xticklabels(fa_t_score.index)
    # plt.show()
    # fa_t_score1=pd.DataFrame()
    # fa_t_score1=fa_t_score.sort_values(by='综合得分',ascending=False).head()
    # ax2 = plt.subplot(111)
    # X1 = fa_t_score1['ID']
    # Y1 = fa_t_score1['综合得分']
    # plt.bar(X1, Y1, color="#87CEFA")
    # # plt.bar(X1, Y1, color='red')
    # plt.title('result01')
    # plt.show()
if __name__ == '__main__':
    main()




