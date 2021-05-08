# coding:utf-8

# 导入NumPy库 
import numpy as np
# 导入画图工具
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score,train_test_split,ShuffleSplit,LeaveOneOut,GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn import metrics

import warnings

#交叉验证法
def cross_validation():
	iris_dataset = datasets.load_iris()
	X,y = datasets.load_iris(return_X_y=True)
	print(X.shape,X.shape)
	X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.4, random_state=0)
	print("X_train,的形态:{}".format(X_train.shape))
	print("X_test的形态:{}".format(X_test.shape))
	print("y_train的形态:{}".format(y_train.shape))
	print("y_test的形态:{}".format(y_test.shape))
	svc = svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
	print('交叉验证法前测试数据的得分：{:.2%}:\n'.format(svc.score(X_test,y_test)))
	svc = svm.SVC(kernel='linear',C=1)
	scores = cross_val_score(svc,X,y,cv=5)#实现交叉验证，cv=5:分5组
	print('交叉验证法后测试数据的得分：{}:\n'.format(scores))
	print('交叉验证法后测试数据的平均分：{:.2%}:\n'.format(scores.mean()))
	X_new =  np.array([[4.5,3.6,1.3,0.3]])
	svc.fit(X_train,y_train)
	prediction = svc.predict(X_new)
	print('预测的鸢尾花为：{}:\n'.format(iris_dataset['target_names'][prediction]))
	########################################################################################
	#随机差分,分为10份
	shuffle_split = ShuffleSplit(test_size=.2,train_size=.7,n_splits=10)
	scores = cross_val_score(svc,X,y,cv=shuffle_split)
	print('随机拆分交叉验证法后测试数据的得分：{}:\n'.format(scores))
	print('随机拆分交叉验证法后测试数据的平均得分：{:.2%}:\n'.format(scores.mean()))
	svc.fit(X_train,y_train)
	prediction = svc.predict(X_new)  
	print('随机拆分预测的鸢尾花为：{}:\n'.format(iris_dataset['target_names'][prediction]))
	########################################################################################
	#挨个试试
	cv = LeaveOneOut()
	scores = cross_val_score(svc,X,y,cv=cv)
	print("迭代次数:{}".format(len(scores)))
	print("挨个试试交叉验证法后测试数据的平均得分:{:.2%}".format(scores.mean()))
	svc.fit(X_train,y_train)
	prediction = svc.predict(X_new)
	print('挨个试试预测的鸢尾花为：{}:\n'.format(iris_dataset['target_names'][prediction]))

def Grid_search():
	warnings.filterwarnings("ignore")
	data = datasets.load_wine()
	X_train,X_test,y_train,y_test = train_test_split(data.data,data.target, random_state=38)
	best_score = 0
	for alpha in [0.01,0.1,1.0,10.0]:
		for max_iter in [10,1000,5000,10000]:
			lasso = Lasso(alpha=alpha,max_iter=max_iter)
			lasso.fit(X_train,y_train)
			score = lasso.score(X_test,y_test)
			if score > best_score:
				best_score = score
				best_params={"alpha":alpha,"最大迭代数":max_iter}
	print("random_state=38,模型最高得分:\n{:.2%}".format(best_score))
	print("random_state=38,最高得分时的参数:\n{}".format(best_params))
	##########################################################################################
	X_train,X_test,y_train,y_test = train_test_split(data.data,data.target, random_state=0)
	best_score = 0
	for alpha in [0.01,0.1,1.0,10.0]:
		for max_iter in [10,1000,5000,10000]:
			lasso = Lasso(alpha=alpha,max_iter=max_iter)
			lasso.fit(X_train,y_train)
			score = lasso.score(X_test,y_test)
			if score > best_score:
				best_score = score
				best_params={"alpha":alpha,"最大迭代数":max_iter}
	print("random_state=0,模型最高得分:\n{:.2%}".format(best_score))
	print("random_state=0,最高得分时的参数:\n{}".format(best_params))
	##########################################################################################
	best_score = 0
	for alpha in [0.01,0.1,1.0,10.0]:
		for max_iter in [10,1000,5000,10000]:
			lasso = Lasso(alpha=alpha,max_iter=max_iter)
			scores = cross_val_score(lasso,X_train,y_train,cv=6)
			score = np.mean(scores)
			if score > best_score:
				best_score = score
				best_params={"alpha":alpha,"最大迭代数":max_iter}
	print("交叉验证与网格搜索模型最高得分:\n{:.2%}".format(best_score))
	print("交叉验证与网格搜索最高得分时的参数:\n{}".format(best_params))
	########################################################################################
	i = 0
	for key,value in best_params.items():
		if i==0:
			alpha = float(value)
		if i==1:
			max_iter = float(value)
		i = i+1
	print("alpha:",alpha)
	print("max_iter:",max_iter)
	lasso = Lasso(alpha=alpha,max_iter=max_iter).fit(X_train,y_train)
	print("最终测试数据得分{:.2%}".format(lasso.score(X_test,y_test)))
	########################################################################################
	#用GridSeearchCV简化
	params = {"alpha":[0.01,0.1,1.0,10.0],"max_iter":[10,1000,5000,10000]}
	gread_search = GridSearchCV(lasso,params,cv=6)
	gread_search.fit(X_train,y_train)
	print("模型最高得分:\n{:.2%}".format(gread_search.score(X_test,y_test)))
	print("最高得分时的参数:\n{}".format(gread_search.best_params_))
	print("交叉验证最高得分:\n{:.2%}".format(gread_search.best_score_))

def accuracy_rate():
	X,y = make_blobs(n_samples=200,centers=2, random_state=1,cluster_std=5) #cluster_std:方差
	plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.cool,edgecolor='k')
	plt.show()
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=68)
	######predict_proba###############################################################
	gnb = GaussianNB()
	gnb.fit(X_train,y_train)
	predict_proba = gnb.predict_proba(X_test)
	print("预测准确率形态{}".format(predict_proba.shape))
	print("预测准确率前5个数据:\n",predict_proba[:5])
	x_min,x_max = X[:,0].min()-0.5,X[:,0].max()+0.5
	y_min,y_max = X[:,1].min()-0.5,X[:,1].max()+0.5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),np.arange(y_min, y_max, .02))
	Z = gnb.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
	Z = Z.reshape(xx.shape)
	#画等高线
	plt.contourf(xx,yy,Z,cmap=plt.cm.summer,alpha=0.8)
	#画散点图
	plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.cool,edgecolors='k')
	plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=plt.cm.cool,edgecolors='k')
	plt.xlim(xx.min(),xx.max()) 
	plt.ylim(yy.min(),yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.show()
	######decision_function###############################################################
	svc = SVC().fit(X_train,y_train)
	dec_func = svc.decision_function(X_test)
	print("决定系数形态{}".format(dec_func.shape))
	print("决定系数前5个数据:\n",dec_func[:5])
	Z = svc.decision_function(np.c_[xx.ravel(),yy.ravel()])
	Z = Z.reshape(xx.shape)
	#画等高线
	plt.contourf(xx,yy,Z,cmap=plt.cm.summer,alpha=0.8)
	#画散点图
	plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=plt.cm.cool,edgecolors='k')
	plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=plt.cm.cool,edgecolors='k')
	plt.xlim(xx.min(),xx.max()) 
	plt.ylim(yy.min(),yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.show()
	
def my_score():
	X,y = datasets.load_wine().data,datasets.load_wine().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=68)
	knn = KNeighborsClassifier()
	knn.fit(X_train, y_train)
	print("训练集得分:\n{:.2%}".format(knn.score(X_train,y_train)))
	print("测试集得分:\n{:.2%}".format(knn.score(X_test,y_test)))
	y_true = y_test
	y_pred = knn.predict(X_test)
	#混淆矩阵
	print("混淆矩阵:\n",confusion_matrix(y_true, y_pred))
	#准确性
	accuracy = '{:.1%}'.format(accuracy_score(y_true, y_pred))
	print("准确性:",accuracy)
	#精确性
	precision = '{:.1%}'.format(precision_score(y_true, y_pred, average='micro'))
	print("精确性:",precision)
	#召回率
	recall = '{:.1%}'.format(recall_score(y_true, y_pred, average='micro'))
	print("召回率:",recall)
	#F1值
	f1score = '{:.1%}'.format(f1_score(y_true, y_pred, average='micro'))
	print("F1值:",f1score)

	X,y = make_blobs(n_samples=200,centers=2, random_state=1,cluster_std=5)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=68)
	gnb = GaussianNB()
	gnb.fit(X_train,y_train)
	predict_proba = gnb.predict_proba(X_test)
	mylist = [None] * int(predict_proba.shape[0])
	i = 0
	for my_predict_proba in predict_proba:
		sign = int(y[i])
		mylist[i] = my_predict_proba[sign]
		i = i+1
	GTlist = y_test
	Problist = mylist
	fpr, tpr, thresholds = roc_curve(GTlist, Problist, pos_label=1)
	roc_auc = metrics.auc(fpr, tpr)  #auc为Roc曲线下的面积
	print("AUC值:",end='')
	print('{:.1%}'.format(roc_auc))
	plt.rcParams['font.sans-serif']=['SimHei']
	plt.rcParams['axes.unicode_minus']=False
	#ROC曲线
	plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	# plt.plot([0, 1], [0, 1], 'r--')
	plt.xlim([-0.1, 1.1])
	plt.ylim([-0.1, 1.1])
	plt.xlabel(u'假阳性率') #横坐标是fpr
	plt.ylabel(u'真阳性率')  #纵坐标是tpr
	plt.title(u'接收器工作特性示例')
	plt.show()

	#P-R曲线 
	plt.figure(u"P-R 曲线")
	plt.title(u'精度/召回曲线')
	plt.xlabel(u'召回')
	plt.ylabel(u'精度')
	#y_true为样本实际的类别，y_scores为样本为正例的概率
	y_true = np.array(GTlist)
	y_scores = np.array(Problist)
	precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
	plt.plot(recall,precision)
	plt.show()
	
def broken_line():
    x = np.linspace(-100,100,10000)
    y = -0.5*x**2+40
    #y_bie = -x+40
    #y_bie = -x=0;x=0,y=40
    y_bie =40+0*x
    #flg =plt.figure(figsize=(8,4))#创建图大小
    plt.xlim(-30,30)#设置x轴上下限
    plt.ylim(-20,100)#设置y轴上下限

    y_line = plt.plot(x,y,color='blue',linestyle='dashdot')
    yy_line = plt.plot(x,y_bie,color='red',linestyle='dashdot')
    plt.plot([0],[40],'o')
    plt.plot(x,y,label='line')

    plt.legend(loc='upper left',frameon=True)
    plt.show()
    
if __name__=="__main__":
	#cross_validation()
	#Grid_search()
	#accuracy_rate()
	#my_score()
        broken_line()




