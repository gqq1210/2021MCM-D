import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from sklearn import svm, linear_model, naive_bayes, ensemble, neighbors, tree
from sklearn.metrics import mean_squared_error

data_music = pd.read_csv('full_music_data.csv')
data_music = data_music.values.tolist()

# job = pd.read_csv('loss-job.csv')
# job = job.values.tolist()
#
# key = []
# value = []
# for i in job[:]:
#     key.append(i[0])
#     value.append((i[1]))
#
# dic = dict(zip(key,value))
#
#
# X = []
# social = []
# for i in data_music:
#     data = i[-2].split('/')
#     if(len(data)>1 and int(data[-1]) >= 1948 and int(data[-1]) <= 2020):
#         if(len(data[0]) == 1):
#             data[0] = '0' + data[0]
#         new = data[-1] + '-' + data[0]
#         X.append(i[2:-2])
#         social.append(dic[new])
#
#     elif(len(data) == 1 and len(data[0])==4 and int(data[0]) >= 1948 and int(data[0]) <= 2020):
#         new = data[0] + '-06'
#         X.append(i[2:-2])
#         social.append(dic[new])
#
#
# print(len(X))
# print(len(social))
# Y = social
#
# X_train = X[:50000]
# Y_train = Y[:50000]
# X_test = X[50000:55000]
# Y_test = Y[50000:55000]

#%%

# #最小二乘法
# clf=linear_model.LinearRegression()
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# plt.scatter(range(0, len(Y_test)), Y_test, color='blue', label="true")
# plt.plot(range(0, y_pred.size), y_pred, color='Red', label="predict")
# print(y_pred)
# mse_predict = mean_squared_error(Y_test, y_pred)
# print("最小二乘法预测误差为：", mse_predict)
# plt.title("LinearRegression")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Social indicators")
# plt.legend(loc='upper right')
# plt.savefig('ques7/LinearRegression.png', bbox_inches="tight", dpi=300)
# plt.show()
# #%%
#
# #岭回归
# clf=linear_model.Ridge()
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# plt.scatter(range(0, len(Y_test)), Y_test, color='blue', label="true")
# plt.plot(range(0, y_pred.size), y_pred, color='Red', label="predict")
# print(y_pred)
# mse_predict = mean_squared_error(Y_test, y_pred)
# print("岭回归预测误差为：", mse_predict)
# plt.title("Ridge")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Social indicators")
# plt.legend(loc='upper right')
# plt.savefig('ques7/Ridge.png', bbox_inches="tight", dpi=300)
# plt.show()
# #%%
#
# #贝叶斯回归
# clf=linear_model.BayesianRidge()
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# plt.scatter(range(0, len(Y_test)), Y_test, color='blue', label="true")
# plt.plot(range(0, y_pred.size), y_pred, color='Red', label="predict")
# print(y_pred)
# mse_predict = mean_squared_error(Y_test, y_pred)
# print("贝叶斯回归预测误差为：", mse_predict)
# plt.title("BayesianRidge")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Social indicators")
# plt.legend(loc='upper right')
# plt.savefig('ques7/BayesianRidge.png', bbox_inches="tight", dpi=300)
# plt.show()
# #%%
#
# #Lasso
# clf=linear_model.Lasso()
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# plt.scatter(range(0, len(Y_test)), Y_test, color='blue', label="true")
# plt.plot(range(0, y_pred.size), y_pred, color='Red', label="predict")
# print(y_pred)
# mse_predict = mean_squared_error(Y_test, y_pred)
# print("Lasso预测误差为：", mse_predict)
# plt.title("Lasso")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Social indicators")
# plt.legend(loc='upper right')
# plt.savefig('ques7/Lasso.png', bbox_inches="tight", dpi=300)
# plt.show()
# #%%
#
# #决策树回归
# from sklearn import tree
# clf=tree.DecisionTreeRegressor()
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# plt.scatter(range(0, len(Y_test)), Y_test, color='blue', label="true")
# plt.plot(range(0, y_pred.size), y_pred, color='Red', label="predict")
# print(y_pred)
# mse_predict = mean_squared_error(Y_test, y_pred)
# print("决策树回归预测误差为：", mse_predict)
# plt.title("DecisionTree")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Social indicators")
# plt.legend(loc='upper right')
# plt.savefig('ques7/DecisionTree.png', bbox_inches="tight", dpi=300)
# plt.show()
# #%%
#
#
# #最近邻回归
# from sklearn import neighbors
# clf=neighbors.KNeighborsRegressor()
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# plt.scatter(range(0, len(Y_test)), Y_test, color='blue', label="true")
# plt.plot(range(0, y_pred.size), y_pred, color='Red', label="predict")
# print(y_pred)
# mse_predict = mean_squared_error(Y_test, y_pred)
# print("最近邻回归预测误差为：", mse_predict)
# plt.title("KNeighbors")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Social indicators")
# plt.legend(loc='upper right')
# plt.savefig('ques7/KNeighbors.png', bbox_inches="tight", dpi=300)
# plt.show()



# #高斯回归
# from sklearn import gaussian_process
# clf=gaussian_process.GaussianProcessRegressor()
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# plt.scatter(range(0, len(Y_test)), Y_test, color='blue', label="true")
# plt.plot(range(0, y_pred.size), y_pred, color='Red', label="predict")
# print(y_pred)
# mse_predict = mean_squared_error(Y_test, y_pred)
# print("高斯回归预测误差为：", mse_predict)
# plt.title("Gaussian")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Social indicators")
# plt.legend(loc='upper right')
# plt.savefig('ques7/GaussianProcess.png', bbox_inches="tight", dpi=300)
# plt.show()
# #%%
#

# #%%

# #人工神经网络
# from sklearn import neural_network
# clf=neural_network.MLPRegressor()
# clf.fit(X_train, Y_train)
# y_pred = clf.predict(X_test)
# plt.scatter(range(0, len(Y_test)), Y_test, color='blue', label="true")
# plt.plot(range(0, y_pred.size), y_pred, color='Red', label="predict")
# print(y_pred)
# mse_predict = mean_squared_error(Y_test, y_pred)
# print("人工神经网络预测误差为：", mse_predict)
# plt.title("Neural Network")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Social indicators")
# plt.legend(loc='upper right')
# plt.savefig('ques7/neural_network.png', bbox_inches="tight", dpi=300)
# plt.show()
# #%%

#误差越小，说明预测准确率越高，因此排序情况为（mse值从小到大排序）
#最小二乘法（10.81） 岭回归（11.30） 贝叶斯回归（12.73） Lasso（13.07） 决策树回归（19.71） 最近邻回归（25.87） 高斯回归（368.67） 人工神经网络（851.63）
# 最小二乘法效果最好，误差最小









X = []
economic = []
for i in data_music[:2000]:
    if (i[-3]==1920 or i[-3] == 1921 or (i[-3]>=1929 and i[-3] <=1933) or (i[-3]>=1937 and i[-3] <=1938) or (i[-3]>=1948 and i[-3] <=1949) or (i[-3]>=1957 and i[-3] <=1958) or (i[-3]>=1969 and i[-3] <=1970) or (i[-3]>=1974 and i[-3] <=1975) or (i[-3]>=1980 and i[-3] <=1982) or (i[-3]>=1990 and i[-3] <=1991) or (i[-3]>=2007 and i[-3] <=2012)):
        i[-2] = 0
    else:
        i[-2] = 3
    X.append(i[2:-2])
    economic.append(i[-2])


clf=svm.SVC(gamma='auto')
clf.fit(X, economic)
y_pred = clf.predict(X)
plt.scatter(range(0, len(economic)), economic, color='blue', marker='|', label ="true")
plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
count=0
kk = 0
for i in range(y_pred.size):
    if y_pred[i] != economic[i]:
        count=count+1
        if(kk==0):
            plt.vlines(i, y_pred[i], economic[i],color="black", label = "error")
            kk=1
        else:
            plt.vlines(i, y_pred[i], economic[i], color="black")
print("error=", count)
plt.title("SVM")
plt.tight_layout()
plt.xticks([])
plt.ylabel("Political era")
plt.legend(loc='upper right')
plt.savefig('ques7/svm.png', bbox_inches="tight", dpi=300)
plt.show()







clf=linear_model.LogisticRegression()
clf.fit(X, economic)
y_pred = clf.predict(X)
plt.scatter(range(0, len(economic)), economic, color='blue', marker='|', label ="true")
plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
count=0
kk = 0
for i in range(y_pred.size):
    if y_pred[i] != economic[i]:
        count=count+1
        if(kk==0):
            plt.vlines(i, y_pred[i], economic[i],color="black", label = "error")
            kk=1
        else:
            plt.vlines(i, y_pred[i], economic[i], color="black")
print("error=", count)
plt.title("LogisticRegression")
plt.tight_layout()
plt.xticks([])
plt.ylabel("Political era")
plt.legend(loc='upper right')
plt.savefig('ques7/logisticRegression.png', bbox_inches="tight", dpi=300)
plt.show()

#%%

clf=tree.DecisionTreeClassifier()
clf.fit(X, economic)
y_pred = clf.predict(X)
plt.scatter(range(0, len(economic)), economic, color='blue', marker='|', label ="true")
plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
count=0
kk = 0
for i in range(y_pred.size):
    if y_pred[i] != economic[i]:
        count=count+1
        if(kk==0):
            plt.vlines(i, y_pred[i], economic[i],color="black", label = "error")
            kk=1
        else:
            plt.vlines(i, y_pred[i], economic[i], color="black")
print("error=", count)
plt.title("DecisionTree")
plt.tight_layout()
plt.xticks([])
plt.ylabel("Political era")
plt.legend(loc='upper right')
plt.savefig('ques7/tree.png', bbox_inches="tight", dpi=300)
plt.show()

# #%%

clf=neighbors.KNeighborsClassifier()
clf.fit(X, economic)
y_pred = clf.predict(X)
plt.scatter(range(0, len(economic)), economic, color='blue', marker='|', label ="true")
plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
count=0
kk = 0
for i in range(y_pred.size):
    if y_pred[i] != economic[i]:
        count=count+1
        if(kk==0):
            plt.vlines(i, y_pred[i], economic[i],color="black", label = "error")
            kk=1
        else:
            plt.vlines(i, y_pred[i], economic[i], color="black")
print("error=", count)
plt.title("KNeighbors")
plt.tight_layout()
plt.xticks([])
plt.ylabel("Political era")
plt.legend(loc='upper right')
plt.savefig('ques7/KNeighbor.png', bbox_inches="tight", dpi=300)
plt.show()

# #%%

clf=ensemble.RandomForestClassifier(n_estimators=20)
clf.fit(X, economic)
y_pred = clf.predict(X)
plt.scatter(range(0, len(economic)), economic, color='blue', marker='|', label ="true")
plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
count=0
kk = 0
for i in range(y_pred.size):
    if y_pred[i] != economic[i]:
        count=count+1
        if(kk==0):
            plt.vlines(i, y_pred[i], economic[i],color="black", label = "error")
            kk=1
        else:
            plt.vlines(i, y_pred[i], economic[i], color="black")
print("error=", count)
plt.title("RandomForest")
plt.tight_layout()
plt.xticks([])
plt.ylabel("Political era")
plt.legend(loc='upper right')
plt.savefig('ques7/Random.png', bbox_inches="tight", dpi=300)
plt.show()

# #%%

clf=naive_bayes.GaussianNB()
clf.fit(X, economic)
y_pred = clf.predict(X)
plt.scatter(range(0, len(economic)), economic, color='blue', marker='|', label ="true")
plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
count=0
kk = 0
for i in range(y_pred.size):
    if y_pred[i] != economic[i]:
        count=count+1
        if(kk==0):
            plt.vlines(i, y_pred[i], economic[i],color="black", label = "error")
            kk=1
        else:
            plt.vlines(i, y_pred[i], economic[i], color="black")
print("error=", count)
plt.title("GaussianNB")
plt.tight_layout()
plt.xticks([])
plt.ylabel("Political era")
plt.legend(loc='upper right')
plt.savefig('ques7/GaussianNB.png', bbox_inches="tight", dpi=300)
plt.show()
















# X = []
# political = []
# for i in data_music[:2000]:
#     if (i[-3]<=1929):
#         i[-2] = 0
#     elif (i[-3]>=1930 and i[-3]<=1945):
#         i[-2] = 1
#     elif (i[-3]>=1946 and i[-3]<=1969):
#         i[-2] = 2
#     else:
#         i[-2] = 3
#     X.append(i[2:-2])
#     political.append(i[-2])

#
# clf=svm.SVC(gamma='auto')
# clf.fit(X, political)
# y_pred = clf.predict(X)
# plt.scatter(range(0, len(political)), political, color='blue', marker='|', label ="true")
# plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
# count=0
# kk = 0
# for i in range(y_pred.size):
#     if y_pred[i] != political[i]:
#         count=count+1
#         if(kk==0):
#             plt.vlines(i, y_pred[i], political[i],color="black", label = "error")
#             kk=1
#         else:
#             plt.vlines(i, y_pred[i], political[i], color="black")
# print("error=", count)
# plt.title("SVM")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Political era")
# plt.legend(loc='upper right')
# plt.savefig('ques7/svm.png', bbox_inches="tight", dpi=300)
# plt.show()
#
#
#
#
#
#
#
# clf=linear_model.LogisticRegression()
# clf.fit(X, political)
# y_pred = clf.predict(X)
# plt.scatter(range(0, len(political)), political, color='blue', marker='|', label ="true")
# plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
# count=0
# kk = 0
# for i in range(y_pred.size):
#     if y_pred[i] != political[i]:
#         count=count+1
#         if(kk==0):
#             plt.vlines(i, y_pred[i], political[i],color="black", label = "error")
#             kk=1
#         else:
#             plt.vlines(i, y_pred[i], political[i], color="black")
# print("error=", count)
# plt.title("LogisticRegression")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Political era")
# plt.legend(loc='upper right')
# plt.savefig('ques7/logisticRegression.png', bbox_inches="tight", dpi=300)
# plt.show()
#
# #%%
#
# clf=tree.DecisionTreeClassifier()
# clf.fit(X, political)
# y_pred = clf.predict(X)
# plt.scatter(range(0, len(political)), political, color='blue', marker='|', label ="true")
# plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
# count=0
# kk = 0
# for i in range(y_pred.size):
#     if y_pred[i] != political[i]:
#         count=count+1
#         if(kk==0):
#             plt.vlines(i, y_pred[i], political[i],color="black", label = "error")
#             kk=1
#         else:
#             plt.vlines(i, y_pred[i], political[i], color="black")
# print("error=", count)
# plt.title("DecisionTree")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Political era")
# plt.legend(loc='upper right')
# plt.savefig('ques7/tree.png', bbox_inches="tight", dpi=300)
# plt.show()
#
# # #%%
#
# clf=neighbors.KNeighborsClassifier()
# clf.fit(X, political)
# y_pred = clf.predict(X)
# plt.scatter(range(0, len(political)), political, color='blue', marker='|', label ="true")
# plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
# count=0
# kk = 0
# for i in range(y_pred.size):
#     if y_pred[i] != political[i]:
#         count=count+1
#         if(kk==0):
#             plt.vlines(i, y_pred[i], political[i],color="black", label = "error")
#             kk=1
#         else:
#             plt.vlines(i, y_pred[i], political[i], color="black")
# print("error=", count)
# plt.title("KNeighbors")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Political era")
# plt.legend(loc='upper right')
# plt.savefig('ques7/KNeighbor.png', bbox_inches="tight", dpi=300)
# plt.show()
#
# # #%%
#
# clf=ensemble.RandomForestClassifier(n_estimators=20)
# clf.fit(X, political)
# y_pred = clf.predict(X)
# plt.scatter(range(0, len(political)), political, color='blue', marker='|', label ="true")
# plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
# count=0
# kk = 0
# for i in range(y_pred.size):
#     if y_pred[i] != political[i]:
#         count=count+1
#         if(kk==0):
#             plt.vlines(i, y_pred[i], political[i],color="black", label = "error")
#             kk=1
#         else:
#             plt.vlines(i, y_pred[i], political[i], color="black")
# print("error=", count)
# plt.title("RandomForest")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Political era")
# plt.legend(loc='upper right')
# plt.savefig('ques7/Random.png', bbox_inches="tight", dpi=300)
# plt.show()
#
# # #%%
#
# clf=naive_bayes.GaussianNB()
# clf.fit(X, political)
# y_pred = clf.predict(X)
# plt.scatter(range(0, len(political)), political, color='blue', marker='|', label ="true")
# plt.scatter(range(0, y_pred.size), y_pred, color='Red', marker='_', label = "predict")
# count=0
# kk = 0
# for i in range(y_pred.size):
#     if y_pred[i] != political[i]:
#         count=count+1
#         if(kk==0):
#             plt.vlines(i, y_pred[i], political[i],color="black", label = "error")
#             kk=1
#         else:
#             plt.vlines(i, y_pred[i], political[i], color="black")
# print("error=", count)
# plt.title("GaussianNB")
# plt.tight_layout()
# plt.xticks([])
# plt.ylabel("Political era")
# plt.legend(loc='upper right')
# plt.savefig('ques7/GaussianNB.png', bbox_inches="tight", dpi=300)
# plt.show()