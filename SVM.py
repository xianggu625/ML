# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs,make_regression
import mglearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,LinearSVR,SVC,SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split
import warnings
from Util import util
import random
import math

def svm_base():
	myutil = util()
	PI = 3.1415926
	x = np.linspace(0,50,100)
	y = 0 * x
	y0 = 0 * x
	y1 = y + 20
	y2 = y - 20
	plt.plot(x,y,c='black',linestyle='dashdot')
	plt.plot(x,y0,c='black')
	plt.plot(x,y1,c='green')
	plt.plot(x,y2,c='green')
	a = np.random.random([50,1])*50
	c = np.random.random([50]).reshape(-1,1)
	b =np.array([[0 for col in range(2)] for row in range(50)])
	for i in range(50):
		b[i][0] = int(a[i])
		b[i][1] = 0
		if int(a[i]) % 2 ==0:
			c[i] = 0
		else:
			c[i] = 1
	plt.scatter(b[:,0],b[:,1],c=c,cmap=plt.cm.spring,s=30)
	title = u"一维空间内的奇偶数是不可以进行线性分割的"
	myutil.show_pic(title)
##########################################################################
	# 分离
	x = np.linspace(0,50,100)
	y = 0 * x
	y0 = 0 * x
	y1 = y + 20
	y2 = y - 20
	plt.plot(x,y,c='black')
	plt.plot(x,y0,c='black',linestyle='dashdot')
	plt.plot(x,y1,c='green')
	plt.plot(x,y2,c='green')
	for i in range(50):
		if c[i] == 0:
			b[i][1] = 20+random.randint(0,40)
		else:
			b[i][1] = -20-random.randint(0,40)
	plt.scatter(b[:,0],b[:,1],c=c,cmap=plt.cm.spring,s=30)
	title = u"分离到二维空间中可以进行线性分割的"
	myutil.show_pic(title)
##########################################################################
	# 旋转
	for i in range(50):
		x = b[i][0]
		y = b[i][1]
		b[i][0] = x*math.cos(PI/4) - y*math.sin(PI/4)
		b[i][1] = x*math.sin(PI/4) + y*math.cos(PI/4)
	plt.scatter(b[:,0],b[:,1],c=c,cmap=plt.cm.spring,s=30)
	x = np.linspace(min(b[:,0])-5,max(b[:,0])+5,100)
	yx = np.linspace(-60,100,100)
	xx = np.linspace(0,0,100)
	y = x
	y0 = 0 * x
	y1 = x + 20
	y2 = x - 20
	plt.plot(x,y,c='black')
	plt.plot(x,y0,c='black',linestyle='dashdot') # X轴
	plt.plot(xx,yx,c='black',linestyle='dashdot')# Y轴
	plt.plot(x,y1,c='green')
	plt.plot(x,y2,c='green')
	title = u"为了更普遍性，进行旋转"
	myutil.show_pic(title)
##########################################################################
	# 移动
	for i in range(50):
		y = b[i][1]
		b[i][1] = y + 40
	plt.scatter(b[:,0],b[:,1],c=c,cmap=plt.cm.spring,s=30)
	x = np.linspace(min(b[:,0])-5,max(b[:,0])+5,100)
	yx = np.linspace(-20,140,100)
	xx = np.linspace(0,0,100)
	y = x+ 40
	y0 = 0 * x
	y1 = x + 20 + 40
	y2 = x - 20 + 40
	plt.plot(x,y,c='black')
	plt.plot(x,y0,c='black',linestyle='dashdot') # X轴
	plt.plot(xx,yx,c='black',linestyle='dashdot')# Y轴
	plt.plot(x,y1,c='green')
	plt.plot(x,y2,c='green')
	title = u"进一步更普遍性，进行平移"
	myutil.show_pic(title)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))
 

def kernel():
        myutil = util()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = np.linspace(-40, 40, 100)
        y = np.linspace(-40, 40, 100)
        z = x*y + 40
        ax.plot(x, y, z, label=u"线性核")
        ax.legend()
        title = u"线性核"
        myutil.show_pic(title)
##############################################################################################################
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = np.linspace(-40, 40, 100)
        y = np.linspace(-40, 40, 100)
        z = (5*x*y+4)**3
        ax.plot(x, y, z, label=u"多项式核")
        ax.legend()
        title = u"多项式核"
        myutil.show_pic(title)
##############################################################################################################
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = np.linspace(-40, 40, 100)
        y = np.linspace(-40, 40, 100)
        z = sigmoid(5*x*y+4)
        ax.plot(x, y, z, label=u"Sigmoid核")
        ax.legend()
        title = u"Sigmoid核"
        myutil.show_pic(title)
##############################################################################################################
	
def using_mglearn():
	mglearn.plots.plot_linear_svc_regularization()
	plt.show()
#####################################################################################################
#SVC
#####################################################################################################
def SVC_Theory():
	myutil = util()
	#创建50个数据点，分成2类
	X , y = make_blobs(n_samples=50,random_state=6,centers=2)
	for kernel in ['linear','rbf','sigmoid','poly']:
		for gamma in ['scale', 'auto',0.1,0.01,0.001]:
			#创建一个线性内核的支持向量
			clf = SVC(kernel=kernel,gamma=gamma,C=1000)# C-SVC的惩罚参数C，默认值是1.0
			clf.fit(X,y)
			# 画出数据点
			plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired,s=30)
			# 建立图像坐标
			ax = plt.gca() #当前的图表和子图可以使用plt.gcf()和plt.gca()获得
			xlim = ax.get_xlim() #返回当前Axes视图的x的上下限
			ylim = ax.get_ylim() #返回当前Axes视图的y的上下限
			# 生成等差数列
			xx = np.linspace(xlim[0],xlim[1],30)
			yy = np.linspace(ylim[0],ylim[1],30)
			YY , XX = np.meshgrid(yy,xx) # meshgrid函数用两个坐标轴上的点在平面上画网格。
			xy = np.vstack([XX.ravel(),YY.ravel()]).T# np.hstack():横向拼接，增加特征量；np.vstack():纵向拼接，增加样本个数
			Z = clf.decision_function(xy). reshape(XX.shape) 
			#decision_function：计算样本点到分割超平面的函数距离
			#shape是查看数据有多少行多少列        #reshape()是数组array中的方法，作用是将数据重新组织
			# 把分类决定边界画出来
			ax.contour(XX,YY,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--']) #绘制等高线
			ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=100,linewidth=1,facecolors='none')
			title=u"VC原理，"+kernel+",gamma="+str(gamma)
			myutil.show_pic(title)

#SVC分析鸢尾花数据
def SVC_for_load_iris():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.load_iris().data,datasets.load_iris().target
	X1 = datasets.load_iris().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	for kernel in ['linear','rbf','sigmoid','poly']:
		for gamma in ['scale', 'auto',0.1,0.01,0.001]:
			clf = SVC(max_iter=100000,kernel=kernel,gamma=gamma)
			clf.fit(X_train, y_train)
			title = "SVC_鸢尾花数据,kernel="+kernel+",gamma="+str(gamma)
			myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
			myutil.plot_learning_curve(SVC(max_iter=100000,kernel=kernel,gamma=gamma),X,y,title)
			myutil.draw_scatter(X,y,clf,title)
			myutil.show_pic(title)
			clf = SVC(max_iter=100000,kernel=kernel,gamma=gamma).fit(X1,y)
			myutil.draw_scatter_for_clf(X1,y,clf,title)

#SVC分析红酒数据
def SVC_for_load_wine():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.load_wine().data,datasets.load_wine().target
	X1 = datasets.load_wine().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	for kernel in ['linear','rbf','sigmoid','poly']:
		for gamma in ['scale', 'auto',0.1,0.01,0.001]:
			clf = SVC(max_iter=100000,kernel=kernel,gamma=gamma)
			clf.fit(X_train, y_train)
			title = "SVC_红酒数据,kernel="+kernel+",gamma="+str(gamma)
			myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
			myutil.plot_learning_curve(SVC(max_iter=100000,kernel=kernel,gamma=gamma),X,y,title)
			myutil.show_pic(title)
			clf = SVC(max_iter=100000,kernel=kernel,gamma=gamma).fit(X1,y)
			myutil.draw_scatter_for_clf(X1,y,clf,title)

#SVC分析乳腺癌数据
def SVC_for_load_breast_cancer():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
	X1 = datasets.load_breast_cancer().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	for kernel in ['linear','rbf','sigmoid','poly']:
		for gamma in ['scale', 'auto',0.1,0.01,0.001]:
			clf = SVC(max_iter=100000,kernel=kernel,gamma=gamma)
			clf.fit(X_train, y_train)
			title = "SVC_乳腺癌数据,kernel="+kernel+",gamma="+str(gamma)
			myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
			myutil.plot_learning_curve(SVC(max_iter=100000,kernel=kernel,gamma=gamma),X,y,title)
			myutil.show_pic(title)
			clf = SVC(max_iter=100000,kernel=kernel,gamma=gamma).fit(X1,y)
			myutil.draw_scatter_for_clf(X1,y,clf,title)
#####################################################################################################
#LinearSVC
#####################################################################################################
def LinearSVC_Theory():
	myutil = util()
	X , y = make_blobs(n_samples=50,random_state=6,centers=2)
	clf = LinearSVC()
	clf.fit(X,y)
	plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired,s=30)
	ax = plt.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	xx = np.linspace(xlim[0],xlim[1],30)
	yy = np.linspace(ylim[0],ylim[1],30)
	YY , XX = np.meshgrid(yy,xx)
	xy = np.vstack([XX.ravel(),YY.ravel()]).T
	Z = clf.decision_function(xy). reshape(XX.shape) 
	ax.contour(XX,YY,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--']) #绘制等高线
	title= u"LinearSVC原理"
	myutil.show_pic(title)

#LinearSVC分析鸢尾花数据
def LinearSVC_for_load_iris():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.load_iris().data,datasets.load_iris().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	clf = LinearSVC()
	clf.fit(X_train, y_train)
	title = "LinearSVC_鸢尾花数据"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(LinearSVC(),X,y,title)
	myutil.show_pic(title)
	X = datasets.load_iris().data[:,:2]
	clf = LinearSVC().fit(X,y)
	myutil.draw_scatter_for_clf(X,y,clf,title)

#LinearSVC分析红酒数据
def LinearSVC_for_load_wine():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.load_wine().data,datasets.load_wine().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	clf = LinearSVC()
	clf.fit(X_train, y_train)
	title = "LinearSVC_红酒数据"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(LinearSVC(),X,y,title)
	myutil.show_pic(title)
	X = datasets.load_wine().data[:,:2]
	clf = LinearSVC().fit(X,y)
	myutil.draw_scatter_for_clf(X,y,clf,title)

#LinearSVC分析乳腺癌数据
def LinearSVC_for_load_breast_cancer():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	clf = LinearSVC()
	clf.fit(X_train, y_train)
	title = "LinearSVC_乳腺癌数据"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.plot_learning_curve(LinearSVC(),X,y,title)
	myutil.show_pic(title)
	X = datasets.load_breast_cancer().data[:,:2]
	clf = LinearSVC().fit(X,y)
	myutil.draw_scatter_for_clf(X,y,clf,title)
#####################################################################################################
#SVR
#####################################################################################################
def SVR_for_make_regression():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = SVR().fit(X,y)
	title = "make_regression SVR()回归线（无噪音）"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(SVR(),X,y,title)
	myutil.show_pic(title)

#加入噪音
def SVR_for_make_regression_add_noise():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = SVR().fit(X,y)
	title = "make_regression SVR()回归线（有噪音）"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(SVR(),X,y,title)
	myutil.show_pic(title)
	
#分析波士顿房价数据的StandardScaler
def SVR_for_boston():
	myutil = util()
	boston = datasets.load_boston()
	X,y = boston.data,boston.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	for kernel in ['linear','rbf','sigmoid','poly']:
		svr = SVR(kernel=kernel)
		svr.fit(X_train,y_train)
		title = "SVR kernel=:"+kernel+"(预处理前)"
		myutil.print_scores(svr,X_train,y_train,X_test,y_test,title)
	#对训练集和测试集数据进行预处理
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaler = scaler.transform(X_train)
	X_test_scaler = scaler.transform(X_test)
	for kernel in ['linear','rbf','sigmoid','poly']:
		svr = SVR(kernel=kernel)
		svr.fit(X_train_scaler,y_train)
		title = "SVR kernel=:"+kernel+"(预处理后)"
		myutil.print_scores(svr,X_train_scaler,y_train,X_test_scaler,y_test,title)

#分析波士顿房价数据
def SVR_for_boston_for_gamma():
	myutil = util()
	boston = datasets.load_boston()
	X,y = boston.data,boston.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaler = scaler.transform(X_train)
	X_test_scaler = scaler.transform(X_test)
	for kernel in ['linear','rbf','sigmoid','poly']:
		for gamma in ['scale', 'auto',0.1,0.01,0.001]:
			svr = SVR(kernel=kernel,gamma=gamma)
			svr.fit(X_train_scaler,y_train)
			title = "SVR kernel=:"+kernel+",gamma="+str(gamma)
			myutil.print_scores(svr,X_train_scaler,y_train,X_test_scaler,y_test,title)

#分析糖尿病数据
def SVR_for_diabetes_for_gamma():
	myutil = util()
	diabetes = datasets.load_diabetes()
	X,y = diabetes.data,diabetes.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	for kernel in ['linear','rbf','sigmoid','poly']:
		for gamma in ['scale', 'auto',0.1,0.01,0.001]:
			svr = SVR(kernel=kernel,gamma=gamma)
			svr.fit(X_train,y_train)
			title = "SVR kernel=:"+kernel+",gamma="+str(gamma)
			myutil.print_scores(svr,X_train,y_train,X_test,y_test,title)
#####################################################################################################
#LinearSVR
#####################################################################################################
def LinearSVR_for_make_regression():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = LinearSVR().fit(X,y)
	title = "make_regression LinearSVR()回归线（无噪音）"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(LinearSVR(),X,y,title)
	myutil.show_pic(title)

#加入噪音
def LinearSVR_for_make_regression_add_noise():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = LinearSVR().fit(X,y)
	title = "make_regression LinearSVR()回归线（有噪音）"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(LinearSVR(),X,y,title)
	myutil.show_pic(title)


#LinearSVR分析波士顿房价数据
def LinearSVR_for_boston():
	warnings.filterwarnings("ignore")
	myutil = util()
	boston = datasets.load_boston()
	X,y = boston.data,boston.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_scaler = scaler.transform(X)
	X_train_scaler = scaler.transform(X_train)
	X_test_scaler = scaler.transform(X_test)
	boston = LinearSVR()
	boston.fit(X_train_scaler,y_train)
	title = "LinearSVR for Boston"
	myutil.print_scores(boston,X_train_scaler,y_train,X_test_scaler,y_test,title)
	myutil.plot_learning_curve(LinearSVR(),X_scaler,y,title)
	myutil.show_pic(title)

#LinearSVR分析糖尿病数据
def LinearSVR_for_diabetes():
	warnings.filterwarnings("ignore")
	myutil = util()
	diabetes = datasets.load_diabetes()
	X,y = diabetes.data,diabetes.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_scaler = scaler.transform(X)
	X_train_scaler = scaler.transform(X_train)
	X_test_scaler = scaler.transform(X_test)
	svr = LinearSVR()
	svr.fit(X_train_scaler,y_train)
	title = "LinearSVR for Diabetes"
	myutil.print_scores(svr,X_train_scaler,y_train,X_test_scaler,y_test,title)
	myutil.plot_learning_curve(LinearSVR(),X_scaler,y,title)
	myutil.show_pic(title)

if __name__=="__main__":
        # svm_base()
        # kernel()
        # using_mglearn()
	# SVC_Theory()
	# SVC_for_load_iris()
	# SVC_for_load_wine()
	# SVC_for_load_breast_cancer()
	# LinearSVC_Theory()
	# LinearSVC_for_load_iris()
	# LinearSVC_for_load_wine()
	# LinearSVC_for_load_breast_cancer()
	# SVR_for_make_regression()
	# SVR_for_make_regression_add_noise()
	# SVR_for_boston()
        # SVR_for_boston_for_gamma()
	# SVR_for_diabetes_for_gamma()
	# SVM_for_all_model()
	# SVM_for_boston()
	# LinearSVR_for_make_regression()
	# LinearSVR_for_make_regression_add_noise()
	# LinearSVR_for_boston()
	LinearSVR_for_diabetes()
	


