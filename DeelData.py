from sklearn.datasets import make_blobs
# 导入画图工具
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Binarizer
# 导入数据划分模块、分为训练集和测试集
from sklearn.model_selection import train_test_split
def my_preprocessing():
    # 产生40个新样本，分成2类，随机生成器的种子为8, 标准差为2
    X,y = make_blobs(n_samples=40,centers=2, random_state=5,cluster_std=2)
    #将数据集用散点图方式进行可视化分析
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.cool)
    plt.show()
    #使用StandardScaler进行处理
    x_1 = StandardScaler().fit_transform(X)
    plt.scatter(x_1[:,0],x_1[:,1],c=y,cmap=plt.cm.cool)
    plt.title("StandardScaler")
    plt.show()
    x_2 = MinMaxScaler().fit_transform(X)
    plt.scatter(x_2[:,0],x_2[:,1],c=y,cmap=plt.cm.cool)
    plt.title("MinMaxScaler")
    plt.show()
    x_3 = RobustScaler().fit_transform(X)
    plt.scatter(x_3[:,0], x_3[:,1],c=y,cmap=plt.cm.cool)
    plt.title("RobustScaler")
    plt.show()
    x_4 = Normalizer().fit_transform(X)
    plt.scatter(x_4[:,0], x_4[:,1],c=y,cmap=plt.cm.cool)
    plt.title("Normalizer")
    plt.show()
    x_5 = MaxAbsScaler().fit_transform(X)
    plt.scatter(x_5[:,0], x_5[:,1],c=y,cmap=plt.cm.cool)
    plt.title("MaxAbsScaler")
    plt.show()
    x_6 = QuantileTransformer().fit_transform(X)
    plt.scatter(x_6[:,0], x_6[:,1],c=y,cmap=plt.cm.cool)
    plt.title("QuantileTransformer")
    plt.show()
    x_7 = Binarizer().fit_transform(X)
    plt.ylim(-0.5,1.5)
    plt.scatter(x_7[:,0], x_7[:,1],c=y,cmap=plt.cm.cool)
    plt.title("Binarizer")
    plt.show()

from sklearn.neural_network import MLPClassifier
from sklearn import datasets
def sklearn_for_Nerver():
	wine = datasets.load_wine()
	X_train,X_test,y_train,y_test = train_test_split(wine.data,wine.target,random_state=62)
	mlp = MLPClassifier(hidden_layer_sizes=[100],max_iter=4000,random_state=62)
	mlp.fit(X_train,y_train)
	print("改造前训练模型得分{:.2%}".format(mlp.score(X_train,y_train)))
	print("改造前测试模型得分{:.2%}".format(mlp.score(X_test,y_test)))
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_train_pp = scaler.transform(X_train)
	X_test_pp = scaler.transform(X_test)
	mlp.fit(X_train_pp,y_train)
	print("改造后训练模型得分{:.2%}".format(mlp.score(X_train_pp,y_train)))
	print("改造后测试模型得分{:.2%}".format(mlp.score(X_test_pp,y_test)))


if __name__=="__main__":
    my_preprocessing()
    sklearn_for_Nerver()
