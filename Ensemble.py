# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor,BaggingClassifier,BaggingRegressor,VotingClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification,make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from Util import util
import warnings
import lightgbm as lgb

##########################################################################################################################
#AdaBoostClassifier
##########################################################################################################################
def iris_of_AdaBoostClassifier():
	myutil = util()
	X,y = datasets.load_iris().data,datasets.load_iris().target
	X1 = datasets.load_iris().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	title = "AdaBoostClassifier鸢尾花数据"
	clf = AdaBoostClassifier(n_estimators=50,random_state=11)
	clf.fit(X_train, y_train)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(AdaBoostClassifier(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)
	clf = AdaBoostClassifier(n_estimators=50,random_state=11).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

def wine_of_AdaBoostClassifier():
	myutil = util()
	X,y = datasets.load_wine().data,datasets.load_wine().target
	X1 = datasets.load_wine().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	title = "AdaBoostClassifier红酒数据"
	clf = AdaBoostClassifier(n_estimators=50,random_state=11)
	clf.fit(X_train, y_train)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(AdaBoostClassifier(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)
	clf = AdaBoostClassifier(n_estimators=50,random_state=11).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

def breast_cancer_of_AdaBoostClassifier():
	myutil = util()
	X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
	X1 = datasets.load_breast_cancer().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	title = "AdaBoostClassifier乳腺癌数据"
	clf = AdaBoostClassifier(n_estimators=50,random_state=11)
	clf.fit(X_train, y_train)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(AdaBoostClassifier(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)
	clf = AdaBoostClassifier(n_estimators=50,random_state=11).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

##########################################################################################################################
#AdaBoostRegressor
##########################################################################################################################
def AdaBoostRegressor_of_make_regression():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = AdaBoostRegressor(n_estimators=50,random_state=11).fit(X,y)
	title = "make_regression AdaBoostRegressor()回归线（有噪音）"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.draw_line(X[:,0],y,clf,title)

def diabetes_of_AdaBoostRegressor():
	myutil = util()
	X,y = datasets.load_diabetes().data,datasets.load_diabetes().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0)
	clf = AdaBoostRegressor(n_estimators=50,random_state=11)
	clf.fit(X_train,y_train)
	title = "AdaBoostRegressor算法分析糖尿病数据"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(AdaBoostRegressor(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)

def boston_of_AdaBoostRegressor():
	myutil = util()
	X,y = datasets.load_boston().data,datasets.load_boston().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0)
	clf = AdaBoostRegressor(n_estimators=50,random_state=11)
	clf.fit(X_train,y_train)
	title = "AdaBoostRegressor算法分析波士顿房价数据"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(AdaBoostRegressor(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)

##########################################################################################################################
#BaggingClassifier
##########################################################################################################################
def iris_of_BaggingClassifier():
	myutil = util()
	X,y = datasets.load_iris().data,datasets.load_iris().target
	X1 = datasets.load_iris().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	title = "BaggingClassifier鸢尾花数据"
	clf = AdaBoostClassifier(n_estimators=50,random_state=11)
	clf.fit(X_train, y_train)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(AdaBoostClassifier(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)
	clf = AdaBoostClassifier(n_estimators=50,random_state=11).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

def wine_of_BaggingClassifier():
	myutil = util()
	X,y = datasets.load_wine().data,datasets.load_wine().target
	X1 = datasets.load_wine().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	title = "BaggingClassifier红酒数据"
	clf = AdaBoostClassifier(n_estimators=50,random_state=11)
	clf.fit(X_train, y_train)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(AdaBoostClassifier(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)
	clf = AdaBoostClassifier(n_estimators=50,random_state=11).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

def breast_cancer_of_BaggingClassifier():
	myutil = util()
	X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
	X1 = datasets.load_breast_cancer().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	title = "BaggingClassifier乳腺癌数据"
	clf = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=4)
	clf.fit(X_train, y_train)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(AdaBoostClassifier(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)
	clf = AdaBoostClassifier(n_estimators=50,random_state=11).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

##########################################################################################################################
#BaggingRegressor
##########################################################################################################################
def BaggingRegressor_of_make_regression():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = BaggingRegressor(n_estimators=50,random_state=11).fit(X,y)
	title = "make_regression BaggingRegressor()回归线（有噪音）"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.draw_line(X[:,0],y,clf,title)

def diabetes_of_BaggingRegressor():
	myutil = util()
	X,y = datasets.load_diabetes().data,datasets.load_diabetes().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0)
	clf = BaggingRegressor(n_estimators=50,random_state=11)
	clf.fit(X_train,y_train)
	title = "BaggingRegressor算法分析糖尿病数据"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(BaggingRegressor(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)

def boston_of_BaggingRegressor():
	myutil = util()
	X,y = datasets.load_boston().data,datasets.load_boston().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0)
	clf = BaggingRegressor(n_estimators=50,random_state=11)
	clf.fit(X_train,y_train)
	title = "BaggingRegressor算法分析波士顿房价数据"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(BaggingRegressor(n_estimators=50,random_state=11),X,y,title)
	myutil.show_pic(title)

##########################################################################################################################
#StackingClassifier
##########################################################################################################################
def My_VotingClassifier(mydata,title):
	warnings.filterwarnings("ignore")
	myutil = util()
	if title=="两个月亮数据":
		X, y = mydata
	else:
		X,y = mydata.data,mydata.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
	votings=['hard','soft']
	for voting in votings:
		voting_clf = VotingClassifier(estimators=[('log_clf', LogisticRegression()),('svm_clf', SVC(probability=True)),('dt_clf', DecisionTreeClassifier(random_state=666))], voting=voting)
		voting_clf.fit(X_train, y_train)
		mytitle = title+" "+voting+"Voting训练"
		myutil.print_scores(voting_clf,X_train,y_train,X_test,y_test,mytitle)
		myutil.plot_learning_curve(VotingClassifier(estimators=[('log_clf', LogisticRegression()),('svm_clf', SVC(probability=True)),('dt_clf', DecisionTreeClassifier(random_state=666))], voting='hard'),X,y,mytitle)
		myutil.show_pic(mytitle)

def call_VotingClassifier():
	mydatas = [datasets.make_moons(n_samples=500, noise=0.3, random_state=42), datasets.load_iris(), datasets.load_wine(), datasets.load_breast_cancer()]
	titles = ["两个月亮数据","鸢尾花数据","红酒数据","乳腺癌数据"]
	for (mydata,title) in zip(mydatas, titles):
		My_VotingClassifier(mydata,title)
        
##########################################################################################################################
#VotingClassifier
##########################################################################################################################
def My_StackingClassifier_1(mydata,title):
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = mydata.data,mydata.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
	#基分类器1：AdaBoostClassifier
	pipe1 = make_pipeline(ColumnSelector(cols=(0, 2)),AdaBoostClassifier())
	#基分类器2：RandomForest
	pipe2 = make_pipeline(ColumnSelector(cols=(1, 2, 3)),RandomForestClassifier())  
	sclf = StackingClassifier(classifiers=[pipe1, pipe2], meta_classifier=LogisticRegression())
	sclf.fit(X_train, y_train)
	mytitle = title+" "+" StackingClassifier"
	myutil.print_scores(sclf,X_train,y_train,X_test,y_test,mytitle)
	myutil.plot_learning_curve(StackingClassifier(classifiers=[pipe1, pipe2], meta_classifier=LogisticRegression()),X,y,mytitle)
	myutil.show_pic(mytitle)

def call_StackingClassifier_1():
	mydatas = [datasets.load_iris(), datasets.load_wine(), datasets.load_breast_cancer()]
	titles = ["鸢尾花数据","红酒数据","乳腺癌数据"]
	for (mydata,title) in zip(mydatas, titles):
		My_StackingClassifier_1(mydata,title)

def My_StackingClassifier_2(mydata,title):
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = mydata.data[:, 1:3],mydata.target
	basemodel1 = AdaBoostClassifier()
	basemodel2 = lgb.LGBMClassifier()
	basemodel3 = RandomForestClassifier(random_state=1)
	lr = LogisticRegression()
	sclf = StackingClassifier(classifiers=[basemodel1, basemodel2, basemodel3], meta_classifier=lr)
	print(title+'五重交叉验证:\n')
	for basemodel, label in zip([basemodel1, basemodel2, basemodel3, sclf], ['adaboost', 'lightgbm', 'Random Forest','StackingClassifier']):
		scores = model_selection.cross_val_score(basemodel,X, y, cv=5, scoring='accuracy')
		print("准确度: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label)) 

def call_StackingClassifier_2():
	mydatas = [datasets.load_iris(), datasets.load_wine(), datasets.load_breast_cancer()]
	titles = ["鸢尾花数据","红酒数据","乳腺癌数据"]
	for (mydata,title) in zip(mydatas, titles):
		My_StackingClassifier_2(mydata,title)
	
if __name__=="__main__":
	#iris_of_AdaBoostClassifier()
	#wine_of_AdaBoostClassifier()
	#breast_cancer_of_AdaBoostClassifier()
	#AdaBoostRegressor_of_make_regression()
	#iris_of_BaggingClassifier()
	#wine_of_BaggingClassifier()
	#breast_cancer_of_BaggingClassifier()
	#BaggingRegressor_of_make_regression()
	#diabetes_of_BaggingRegressor()
	#boston_of_BaggingRegressor()
	#call_VotingClassifier()
	#call_StackingClassifier_1()
	call_StackingClassifier_2()

