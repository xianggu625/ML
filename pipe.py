# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split,learning_curve,ShuffleSplit
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,SVR,LinearSVC,LinearSVR
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor,BaggingClassifier,BaggingRegressor,VotingClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.decomposition import PCA,NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import statsmodels.api as sm
import warnings
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,Normalizer,MaxAbsScaler,QuantileTransformer,Binarizer
from PIL import Image
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap
from mttkinter import mtTkinter as tk
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

def my_piple():
	X,y = make_blobs(n_samples=200,centers=2,cluster_std=5)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=38)
	scaler = StandardScaler().fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	print("训练集形态:",X_train_scaled.shape)
	print("测试集形态:",X_test_scaled.shape)
	#原始的训练集
	plt.scatter(X_train[:,0],X_train[:,1])
	#经过预处理的训练集
	plt.scatter(X_train_scaled[:,0],X_train_scaled[:,1],marker='^',edgecolor='k')
	plt.title(u"训练集 VS 处理后的训练集")
	plt.rcParams['font.sans-serif']=['SimHei']
	plt.rcParams['axes.unicode_minus']=False
	plt.show()
	######################################################################
	params = {'hidden_layer_sizes':[(50,),(100,),(100,100)],"alpha":[0.0001,0.001,0.01]}
	grid = GridSearchCV(MLPClassifier(max_iter=1600,random_state=38),param_grid=params,cv=3)
	grid.fit(X_train_scaled,y_train)
	print("模型最高得分：\n{:.2%}".format(grid.best_score_))
	print("模型最高得分时的参数：\n{}".format(grid.best_params_))
	######################################################################
	#打印模型在测试集上的得分
	print("测试集得分：\n{:.2%}".format(grid.score(X_test_scaled,y_test)))
	######################################################################
	pipeline = Pipeline([('scaler',StandardScaler()),
			     ('mlp',MLPClassifier(max_iter=1600,random_state=38))])
	pipeline.fit(X_train,y_train)
	print("使用管道后的测试集得分：\n{:.2%}".format(pipeline.score(X_test,y_test)))
	######################################################################
	params = {'mlp__hidden_layer_sizes':[(50,),(100,),(100,100)],"mlp__alpha":[0.0001,0.001,0.01]}
	grid = GridSearchCV(pipeline,param_grid=params,cv=3)
	grid.fit(X_train,y_train)
	print("交叉验证最高得分：\n{:.2%}".format(grid.best_score_))
	print("模型最优参数：\n{}".format(grid.best_params_))
	print("测试集得分：\n{:.2%}".format(grid.score(X_test,y_test)))

	
def stock():
	stock = pd.read_csv('stock.csv',encoding='GBK')
	X = stock.loc[:,'价格':'流通市值']
	y = stock['涨跌幅']
	warnings.filterwarnings("ignore")
	#使用管道，Pipeline()方法与make_pipeline()等同
	pipeline = Pipeline([('scaler',StandardScaler()),
			     ('mlp',MLPRegressor(max_iter=1600,hidden_layer_sizes=[1,1],random_state=6))])
	pipe = make_pipeline(StandardScaler(),MLPRegressor(max_iter=1600,hidden_layer_sizes=[1,1],random_state=6))
	scores = cross_val_score(pipe,X,y,cv=20)
	print("pipe处理后模型平均分：{:.2%}".format(float(scores.mean())))
	###########################################################################
	pipe = make_pipeline(StandardScaler(),SelectFromModel(RandomForestRegressor(random_state=6)),
					     MLPRegressor(max_iter=1600,hidden_layer_sizes=[1,1],random_state=6))
	scores = cross_val_score(pipe,X,y,cv=20)
	print("经过pipe处理后，再经过SelectFromModel处理，模型平均分：{:.2%}".format(float(scores.mean())))
	###########################################################################
	params =[{'reg':[MLPRegressor(max_iter=1600,hidden_layer_sizes=[1,1],random_state=6)],
		  'scaler':[StandardScaler(),None]},
		 {'reg':[RandomForestRegressor(random_state=6)],
		  'scaler':[None]}]
	pipe = Pipeline([('scaler',StandardScaler()),('reg',MLPRegressor())])
	grid = GridSearchCV(pipe,params,cv=6)
	grid.fit(X,y)
	print("GridSearchCV处理后，最佳模型是：{}".format(grid.best_params_))
	print("GridSearchCV处理后，模型最佳得分：{:.2%}".format(grid.best_score_))
	###########################################################################
	params =[{'reg':[MLPRegressor(max_iter=1600,random_state=6)],
		  'scaler':[StandardScaler(),None],
		  'reg__hidden_layer_sizes':[(1),(50,),(100,),(1,1),(50,50),(100,100)]},
		 {'reg':[RandomForestRegressor(random_state=6)],
		  'scaler':[None],
		  'reg__n_estimators':[10,50,100]}]
	pipe = Pipeline([('scaler',StandardScaler()),('reg',MLPRegressor())])
	grid = GridSearchCV(pipe,params,cv=6)
	grid.fit(X,y)
	print("加入参数后，最佳模型是：{}".format(grid.best_params_))
	print("加入参数后，模型最佳得分：{:.2%}".format(grid.best_score_))
	###########################################################################
	params =[{'reg':[MLPRegressor(max_iter=1600,random_state=6)],
		  'scaler':[StandardScaler(),None],
		  'reg__hidden_layer_sizes':[(1),(50,),(100,),(1,1),(50,50),(100,100)]},
		 {'reg':[RandomForestRegressor(random_state=6)],
		  'scaler':[None],
		  'reg__n_estimators':[100,500,1000]}]
	pipe = Pipeline([('scaler',StandardScaler()),('reg',MLPRegressor())])
	grid = GridSearchCV(pipe,params,cv=6)
	grid.fit(X,y)
	print("加入参数后，最佳模型是：{}".format(grid.best_params_))
	print("加入参数后，模型最佳得分：{:.2%}".format(grid.best_score_))


def get_better_score():
	warnings.filterwarnings("ignore")
	n_jobs = 2
	params=[{'reg':[LinearRegression()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__n_jobs":[n_jobs]},
				    {'reg':[LogisticRegression()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__n_jobs":[n_jobs]},
				    {'reg':[Ridge()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__alpha":[1,0.1,0.001,0.0001]},
				    {'reg':[Lasso()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__alpha":[1,0.1,0.001,0.0001]},
				    {'reg':[ElasticNet()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__alpha":[0.1,0.5,1,5,10],"reg__l1_ratio":[0.1,0.5,0.9]},
				    {'reg':[RandomForestClassifier()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__n_estimators":[4,5,6,7],"reg__random_state":[2,3,4,5],"reg__n_jobs":[n_jobs],"reg__random_state":[range (0,200)]},
				    {'reg':[RandomForestRegressor()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__n_estimators":[4,5,6,7],"reg__random_state":[2,3,4,5],"reg__n_jobs":[n_jobs],"reg__random_state":[range (0,200)]},
				    {'reg':[DecisionTreeClassifier()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__max_depth":[1,3,5,7],"reg__random_state":[range (1,200)]},
				    {'reg':[DecisionTreeRegressor()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__max_depth":[1,3,5,7],"reg__random_state":[range (1,200)]},
				    {'reg':[KNeighborsClassifier()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__n_jobs":[n_jobs]},
				    {'reg':[KNeighborsRegressor()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__n_jobs":[n_jobs]},
				    {'reg':[BernoulliNB()],'scaler':[StandardScaler(),MinMaxScaler(),None]},
				    {'reg':[GaussianNB()],'scaler':[StandardScaler(),MinMaxScaler(),None]},
				    {'reg':[MultinomialNB()],'scaler':[MinMaxScaler()]},
				    {'reg':[SVC(max_iter=10000)],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__kernel":["linear","rbf","sigmoid","poly"],"reg__gamma":[0.01,0.1,1,5,10],"reg__C":[1.0,3.0,5.0]},
				    {'reg':[SVR(max_iter=100000)],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__kernel":["linear","rbf","sigmoid","poly"],"reg__gamma":[0.01,0.1,1,5,10],"reg__C":[1.0,3.0,5.0]},
				    {'reg':[LinearSVC(max_iter=100000)],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__C":[1.0,3.0,5.0]},
				    {'reg':[LinearSVR(max_iter=100000)],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__C":[1.0,3.0,5.0]},
				    {'reg':[AdaBoostClassifier()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__random_state":[range (1,200)]},
				    {'reg':[AdaBoostRegressor()],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__random_state":[range (1,200)]},
				    {'reg':[VotingClassifier(estimators=[('log_clf', LogisticRegression()),('svm_clf', SVC(probability=True)),('dt_clf', DecisionTreeClassifier(random_state=666))])],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__voting":["hard","soft"],"reg__n_jobs":[n_jobs]},
				    {'reg':[LinearDiscriminantAnalysis(n_components=2)],'scaler':[StandardScaler(),MinMaxScaler(),None]},
				    {'reg':[MLPClassifier(max_iter=100000)],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__activation":["relu","tanh","identity","logistic"],"reg__alpha":[0.0001,0.001,0.01,1],"reg__hidden_layer_sizes":[(1),(50,),(100,),(1,1),(50,50),(100,100)]},
				    {'reg':[MLPRegressor(max_iter=100000)],'scaler':[StandardScaler(),MinMaxScaler(),None],"reg__activation":["relu","tanh","identity","logistic"],"reg__alpha":[0.0001,0.001,0.01,1],"reg__hidden_layer_sizes":[(1),(50,),(100,),(1,1),(50,50),(100,100)]}
				    ]
	stock = pd.read_csv('stock1.csv',encoding='GBK')
	X = stock.loc[:,'价格':'流通市值']
	y = stock['涨跌幅']
	pipe = Pipeline([('scaler',StandardScaler()),('reg',MLPRegressor())])
	shuffle_split = ShuffleSplit(test_size=.2,train_size=.7,n_splits=10)
	grid = GridSearchCV(pipe,params,cv=shuffle_split)
	grid.fit(X,y)
	print("最佳模型是：{}".format(grid.best_params_))
	print("模型最佳训练得分：{:.2%}".format(grid.best_score_))
	print("模型最佳测试得分：{:.2%}".format(grid.score(X,y)))

	
if __name__=="__main__":
	#my_piple()
	#stock()
	get_better_score()



