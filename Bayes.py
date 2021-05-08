# coding:utf-8

# 导入NumPy库 
import numpy as np
# 导入画图工具
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import warnings
from Util import util

def bayes_base():
	X = np.array([[1,1,0,1],[0,1,1,0],[1,0,0,1],[0,1,0,0],[1,0,1,0],[0,1,0,0],[0,0,1,0]])
	y = np.array([1,1,0,0,1,0,1])
	counts={}
	for label in np.unique(y):
		counts[label] = X[y==label].sum(axis=0)
	print("特性统计:\n{}".format(counts))
	clf = BernoulliNB()
	clf.fit(X,y)
	#明天多云
	Next_Day = [[0,0,1,0]]
	pre1 = clf.predict(Next_Day)
	print(pre1)
	#另一天刮风、闷热、预报有雨
	Another_Day = [[1,1,0,1]]
	pre2 = clf.predict(Another_Day)
	print(pre2)
	print(clf.predict_proba(Next_Day))
	print(clf.predict_proba(Another_Day))

############################################################################################
#贝努利贝叶斯
############################################################################################
def bernoulliNB_for_make_blobs():
	myutil = util()
	X,y = make_blobs(n_samples=500,centers=8, random_state=8)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	nb = BernoulliNB()
	nb.fit(X,y)
	title = "贝努利贝叶斯 make_blobs"
	myutil.draw_scatter(X,y,nb,title)
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(BernoulliNB(),X,y,title)
	myutil.show_pic(title)

def bernoulliNB_for_iris():
	myutil = util()
	X,y = datasets.load_iris().data,datasets.load_iris().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	nb = BernoulliNB()
	nb.fit(X,y)
	title = "贝努利贝叶斯 鸢尾花"
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(BernoulliNB(),X,y,title)
	myutil.show_pic(title)

def bernoulliNB_for_wine():
	myutil = util()
	X,y = datasets.load_wine().data,datasets.load_wine().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	nb = BernoulliNB()
	nb.fit(X,y)
	title = "贝努利贝叶斯 红酒"
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(BernoulliNB(),X,y,title)
	myutil.show_pic(title)
	
def bernoulliNB_for_breast_cancer():
	myutil = util()
	X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	nb = BernoulliNB()
	nb.fit(X,y)
	title = "贝努利贝叶斯 乳腺癌"
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(BernoulliNB(),X,y,title)
	myutil.show_pic(title)

############################################################################################
#高斯贝叶斯
############################################################################################
def gaussianNB_for_make_blobs():
	myutil = util()
	X,y = make_blobs(n_samples=500,centers=8, random_state=8)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	nb = GaussianNB()
	nb.fit(X,y)
	title = "高斯贝叶斯 make_blobs"
	myutil.draw_scatter(X,y,nb,title)
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(GaussianNB(),X,y,title)
	myutil.show_pic(title)

def gaussianNB_for_iris():
	myutil = util()
	X,y = datasets.load_iris().data,datasets.load_iris().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	nb = GaussianNB()
	nb.fit(X,y)
	title = "高斯贝叶斯 鸢尾花"
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(GaussianNB(),X,y,title)
	myutil.show_pic(title)

def gaussianNB_for_wine():
	myutil = util()
	X,y = datasets.load_wine().data,datasets.load_wine().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	nb = GaussianNB()
	nb.fit(X,y)
	title = "高斯贝叶斯 红酒"
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(GaussianNB(),X,y,title)
	myutil.show_pic(title)
	
def gaussianNB_for_breast_cancer():
	myutil = util()
	X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	nb = GaussianNB()
	nb.fit(X,y)
	title = "高斯贝叶斯 乳腺癌"
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(GaussianNB(),X,y,title)
	myutil.show_pic(title)

############################################################################################
#多项式贝叶斯
############################################################################################
def multinomialNB_for_make_blobs():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = make_blobs(n_samples=500,random_state=8,centers=8)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	nb = MultinomialNB()
	nb.fit(X_train,y_train)
	title = "多项式贝叶斯 make_blobs"
	myutil.draw_scatter(X,y,nb,title)
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(MultinomialNB(),X,y,title)
	myutil.show_pic(title)

def multinomialNB_for_iris():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.load_iris().data,datasets.load_iris().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	nb = MultinomialNB()
	nb.fit(X_train,y_train)
	title = "多项式贝叶斯 鸢尾花"
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(MultinomialNB(),X,y,title)
	myutil.show_pic(title)

def multinomialNB_for_wine():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.load_wine().data,datasets.load_wine().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	nb = MultinomialNB()
	nb.fit(X_train,y_train)
	title = "多项式贝叶斯 红酒"
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(MultinomialNB(),X,y,title)
	myutil.show_pic(title)
	
def multinomialNB_for_breast_cancer():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
	scaler = MinMaxScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	nb = MultinomialNB()
	nb.fit(X_train,y_train)
	title = "多项式贝叶斯 乳腺癌"
	myutil.print_scores(nb,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(MultinomialNB(),X,y,title)
	myutil.show_pic(title)


if __name__=="__main__":
	#bayes_base()
	#bernoulliNB_for_make_blobs()
	#bernoulliNB_for_iris()
	#bernoulliNB_for_wine()
	#bernoulliNB_for_breast_cancer()
	#gaussianNB_for_make_blobs()
	#gaussianNB_for_iris()
	#gaussianNB_for_wine()
	#gaussianNB_for_breast_cancer()
        #multinomialNB_for_make_blobs()
        #multinomialNB_for_iris()
        #multinomialNB_for_wine()
        multinomialNB_for_breast_cancer()
	#gaussianNB_for_breast_cancer()
	#learning_curve_for_gaussianNB()


