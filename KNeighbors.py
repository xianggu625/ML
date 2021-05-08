# coding:utf-8

# 导入数据集生成器
from sklearn.datasets import make_blobs,make_regression
# 导入KNN分类器
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
# 导入画图工具
import matplotlib.pyplot as plt
# 导入数据划分模块、分为训练集和测试集
from sklearn.model_selection import train_test_split
# 导入NumPy库 
import numpy as np
from sklearn import datasets
from Util import util

def sklearn_Kneighbors_for_make_blobs_2_centers():
	myutil = util()
	# 产生200个新样本，分成2类
	data = make_blobs(n_samples=200,centers=2, random_state=8)
	X,y =data
	print("X is :",X)
	print("y is :",y)
	#将数据集用散点图方式进行可视化分析
	plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolor='k')
	plt.show()    
	clf = KNeighborsClassifier()
	clf.fit(X,y)
	#下面代码用于画图
	title = "KN邻近分类——2个类别"
	myutil.draw_scatter(X,y,clf,title)
	myutil.plot_learning_curve(KNeighborsClassifier(),X,y,title)
	myutil.show_pic(title)

def sklearn_Kneighbors_for_make_blobs_5_centers():
	myutil = util()
	X,y = make_blobs(n_samples=500,centers=5, random_state=8)
	plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolor='k')
	plt.show()
	clf = KNeighborsClassifier()
	clf.fit(X,y)
	#下面代码用于画图
	title = "KN邻近分类——5个类别"
	myutil.draw_scatter(X,y,clf,title)
	myutil.plot_learning_curve(KNeighborsClassifier(),X,y,title)
	myutil.show_pic(title)

def KNeighborsClassifier_for_load_iris():
	myutil = util()
	X,y = datasets.load_iris().data,datasets.load_iris().target
	X1 = datasets.load_iris().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	title = "KNN分类_鸢尾花数据"
	clf = KNeighborsClassifier()
	clf.fit(X_train, y_train)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(KNeighborsClassifier(),X,y,title)
	myutil.show_pic(title)
	clf = KNeighborsClassifier().fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

def KNeighborsClassifier_for_load_wine():
	myutil = util()
	wine_dataset = datasets.load_wine()
	X,y = wine_dataset['data'],wine_dataset['target']
	X1 = datasets.load_wine().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	title = "KNN分类_红酒数据"
	clf = KNeighborsClassifier()
	clf.fit(X_train, y_train)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(KNeighborsClassifier(),X,y,title)
	myutil.show_pic(title)
	X_new =  np.array([[25.5,3.14,3.22,18.5,95.8, 0.97, 2.52, 0.67, 1.52, 7.3, 0.98, 2.96, 990]])
	prediction = clf.predict(X_new)
	print('预测的红酒为：{}:\n'.format(wine_dataset['target_names'][prediction]))
	clf = KNeighborsClassifier().fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

def KNeighborsClassifier_for_load_breast_cancer():
	myutil = util()
	wine_dataset = datasets.load_breast_cancer()
	X,y = wine_dataset['data'],wine_dataset['target']
	X1 = datasets.load_breast_cancer().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	title = "KNN分类_乳腺癌数据"
	clf = KNeighborsClassifier()
	clf.fit(X_train, y_train)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(KNeighborsClassifier(),X,y,title)
	myutil.show_pic(title)
	print('第310个样本预测结果: {}'.format(clf.predict([X[310]])))
	clf = KNeighborsClassifier().fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

def KNeighborsRegressor_for_make_regression():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = KNeighborsRegressor().fit(X,y)
	title = "K邻近算法分析make_regression数据集(无噪音)"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(KNeighborsRegressor(),X,y,title)
	myutil.show_pic(title)

def KNeighborsRegressor_for_make_regression_add_noise():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = KNeighborsRegressor().fit(X,y)
	title = "K邻近算法分析make_regression数据集(有噪音)"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(KNeighborsRegressor(),X,y,title)
	myutil.show_pic(title)

def KNeighborsRegressor_for_for_diabetes():
	myutil = util()
	X,y = datasets.load_diabetes().data,datasets.load_diabetes().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = KNeighborsRegressor().fit(X_train,y_train)
	title = "K邻近回归算法分析糖尿病数据"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(KNeighborsRegressor(),X,y,title)
	myutil.show_pic(title)

def KNeighborsRegressor_for_for_boston():
	myutil = util()
	X,y = datasets.load_boston().data,datasets.load_boston().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = KNeighborsRegressor().fit(X_train,y_train)
	title = "K邻近回归算法分析波士顿房价病数据"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(KNeighborsRegressor(),X,y,title)
	myutil.show_pic(title)

if __name__=="__main__":
	#sklearn_Kneighbors_for_make_blobs_2_centers()
	#sklearn_Kneighbors_for_make_blobs_5_centers()
	KNeighborsClassifier_for_load_iris()
	KNeighborsClassifier_for_load_wine()
	KNeighborsClassifier_for_load_breast_cancer()
	#KNeighborsRegressor_for_make_regression()
	#KNeighborsRegressor_for_make_regression_add_noise()
	#KNeighborsRegressor_for_for_diabetes()
	#KNeighborsRegressor_for_for_boston()
	# sklearn_regression()
	# Sklean_wine()
	# Sklean_iris()
	# Sklean_iris_cross_validation()
	# KNN_for_all_data_and_model()

