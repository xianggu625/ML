# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,learning_curve,ShuffleSplit
from Util import util
from sklearn.datasets import make_regression,make_classification
from sklearn import datasets
import mglearn
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,Ridge,LogisticRegression,Lasso,ElasticNet
from sklearn.preprocessing import StandardScaler 
##############################################
#                   基本概念
##############################################
# 已知直线斜率和截距，画出直线
def Line_base_by_k_b():
	x = np.linspace(-5,5,100)
	y = 0.5 * x + 3
	plt.plot(x,y,c='green')
	plt.title('Straight Line')
	plt.show()

# 通过两个点（[2,3]和[3,4]）画出直线
def Line_base_by_two_point():
	X = [[2],[3]]
	y = [3,4]
	# 用线性模型拟合这；两个点
	lr = LinearRegression().fit(X,y)
	# 画出通过两个点（[2,3]和[3,4]）直线
	z = np.linspace(-5,5,20)
	plt.scatter(X,y,s=80)
	plt.plot(z,lr.predict(z.reshape(-1,1)),c='k')
	plt.title('Straight Line')
	plt.show()
	# 显示这条线的斜率和截距
	print('y={:.3f}'.format(lr.coef_[0]),'x','+{:.3f}'.format(lr.intercept_))

# 画出通过三个点（[2,3]、[3,4]和[4,4]）直线
def Line_base_by_three_point():
	X = [[2],[3],[4]]
	y = [3,4,4]
	# 用线性模型拟合这；两个点
	lr = LinearRegression().fit(X,y)
	# 画出通过三个点（[2,3]、[3,4]和[4,4]）直线
	z = np.linspace(-5,5,20)
	plt.scatter(X,y,s=80)
	plt.plot(z,lr.predict(z.reshape(-1,1)),c='k')
	plt.title('Straight Line')
	plt.show()
	# 显示这条线的斜率和截距
	print('y={:.3f}'.format(lr.coef_[0]),'x','+{:.3f}'.format(lr.intercept_))

#导入make_regression数据集成生成器
# 画出多个点的直线
def Line_base_by_multiple_point():
	X,y = make_regression(n_samples=50,n_features=1,noise=50,n_informative=1,random_state=1)
	# 使用线性模型进行拟合
	reg = LinearRegression()
	reg.fit(X,y)
	# z 是我们生成的等差数列，用来画出线性模型
	z = np.linspace(-3,3,200).reshape(-1,1)
	plt.scatter(X,y,c='b',s=60)
	plt.plot(z,reg.predict(z),c='k')
	plt.title('Linear Regression')
	plt.show()
	# 显示这条线的斜率和截距
	print('y={:.3f}'.format(reg.coef_[0]),'x','+{:.3f}'.format(reg.intercept_))

##############################################
#                   LinearRegression
##############################################
# 导入数据划分模块、分为训练集和测试集
def LinearRegression_for_make_regression():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = LinearRegression().fit(X,y)
	print('lr.coef_: {} '.format(clf.coef_[:]))
	print('reg.intercept_: {}'.format(clf.intercept_))
	print('训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression LinearRegression()回归线（无噪音）"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(LinearRegression(),X,y,title)
	myutil.show_pic(title)

#加入噪音
def LinearRegression_for_make_regression_add_noise():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = LinearRegression().fit(X,y)
	print('lr.coef_: {} '.format(clf.coef_[:]))
	print('reg.intercept_: {}'.format(clf.intercept_))
	print('训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression LinearRegression()回归线（有噪音）"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(LinearRegression(),X,y,title)
	myutil.show_pic(title)

#用线性回归对sklearn糖尿病数据进行分析
def LinearRegression_for_diabetes():
	myutil = util()
	X,y = datasets.load_diabetes().data,datasets.load_diabetes().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8)
	clf = LinearRegression()
	clf = clf.fit(X_train,y_train)
	print('斜率: {} '.format(clf.coef_[:]))
	print('截距: {:.3f}'.format(clf.intercept_))
	print('糖尿病训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('糖尿病测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression 糖尿病数据"
	myutil.plot_learning_curve(LinearRegression(),X,y,title)
	myutil.show_pic(title)

#用线性回归对sklearn波士顿房价数据进行分析
def LinearRegression_for_boston():
	myutil = util()
	Boston = datasets.load_boston()
	X,y = Boston.data,Boston.target
	X_train,X_test,y_train,y_test = train_test_split(X, y)
	clf = LinearRegression().fit(X_train,y_train)
	print('斜率: {} '.format(clf.coef_[:]))
	print('截距: {}'.format(clf.intercept_))
	print('波士顿房价训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('波士顿房价测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression 波士顿房价数据"
	myutil.plot_learning_curve(LinearRegression(),X,y,title)
	myutil.show_pic(title)

def using_mglearn():
	mglearn.plots.plot_linear_regression_wave()
	plt.show()
##############################################
#                   LogisticRegression
##############################################
#LogisticRegression分析make_blobs数据
def LogisticRegression_for_make_blobs():
	myutil = util()
	X,y = make_blobs(n_samples=500,centers=5, random_state=8)
	clf = LogisticRegression(max_iter=100000)
	clf.fit(X,y)
	print('模型正确率：{:.2%}'.format(clf.score(X,y)))
	title = "逻辑回归_make_blobs"
	myutil.draw_scatter(X,y,clf,title)
	myutil.plot_learning_curve(LogisticRegression(max_iter=100000),X,y,title)
	myutil.show_pic(title)

#LogisticRegression分析乳腺癌数据
def LogisticRegression_for_load_breast_cancer():
	myutil = util()
	X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
	X1 = datasets.load_breast_cancer().data[:,:2]
	print("X的shape={},正样本数:{},负样本数:{}".format(X.shape, y[y == 1].shape[0], y[y == 0].shape[0]))
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	# 训练模型
	clf = LogisticRegression(max_iter=100000)
	clf.fit(X_train, y_train)
	# 查看模型得分
	train_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	print("乳腺癌训练集得分:{trs:.2%},乳腺癌测试集得分:{tss:.2%}".format(trs=train_score, tss=test_score))
	title = "逻辑回归_乳腺癌数据"
	myutil.plot_learning_curve(LogisticRegression(max_iter=100000),X,y,title)
	myutil.show_pic(title)
	clf = LogisticRegression(max_iter=100000).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

#LogisticRegression分析鸢尾花数据
def LogisticRegression_for_load_iris():
	myutil = util()
	X,y = datasets.load_iris().data,datasets.load_iris().target
	X1 = datasets.load_iris().data[:,:2]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	clf = LogisticRegression(max_iter=100000)
	clf.fit(X_train, y_train)
	train_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	print("鸢尾花训练集得分:{trs:.2%},鸢尾花测试集得分:{tss:.2%}".format(trs=train_score, tss=test_score))
	title = "逻辑回归_鸢尾花数据"
	myutil.plot_learning_curve(LogisticRegression(max_iter=100000),X,y,title)
	myutil.show_pic(title)
	clf = LogisticRegression(max_iter=100000).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

#LogisticRegression分析红酒数据
def LogisticRegression_for_load_wine():
	myutil = util()
	X,y = datasets.load_wine().data,datasets.load_wine().target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	X1 = datasets.load_wine().data[:,:2]
	clf = LogisticRegression(max_iter=100000)
	clf.fit(X_train, y_train)
	train_score = clf.score(X_train, y_train)
	test_score = clf.score(X_test, y_test)
	print("红酒训练集得分:{trs:.2%},红酒测试集得分:{tss:.2%}".format(trs=train_score, tss=test_score))
	title = "逻辑回归_红酒数据"
	myutil.plot_learning_curve(LogisticRegression(max_iter=100000),X,y,title)
	myutil.show_pic(title)
	clf = LogisticRegression(max_iter=100000).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,clf,title)

##############################################
#                   StatsModels
##############################################
# 导入StatsModels的API库,C:\Users\xiang>pip3 install statsmodels
#StatsModels 库
# y = w^x+e(e 误差，符合均值为0的正态分布)
def StatsModels_linear_regression():
    # 前四行训练构造函数，自变量x 因变量y
    # 通过自变量x准备数据，将1~10 数据分割成20份
    x = np.linspace(0,10,20)
    # 向数组中添加一列，构成20组x
    X = sm.add_constant(x)
    ratio = np.array([1,10]) # ratio：比例
    # 使回归方程的系数点乘x数据集，构成因变量y
    # numpy.random.normal(loc=0.0, scale=1.0, size=None)
    # loc:float,概率分布的均值，对应着整个分布的中心center
    # scale: 概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高
    # size: 输出的shape，默认为None，只输出一个值
    # np.random.randn(size)所谓标准正太分布（μ=0, σ=1），对应于np.random.normal(loc=0, scale=1, size)
    y = np.dot(X,ratio) + np.random.normal(size=20)
    # 执行OLS模型
    testmodel = sm.OLS(y,X)
    # 使用fit方法启动模型训练
    results = testmodel.fit()
    print("results.params:\n",results.params, "\nresults.summary():\n",results.summary())
    
def StatsModels_linear_regression_for_boston_data():
	#Load Boston
	Boston = datasets.load_boston()
	X,y = Boston.data,Boston.target
	testmodel = sm.OLS(y,X)
	results = testmodel.fit()
	print("results.params(Boston):\n",results.params, "\nresults.summary(Boston):\n",results.summary())
	#Load diabetes
	X,y = datasets.load_diabetes().data,datasets.load_diabetes().target
	testmodel = sm.OLS(y,X)
	results = testmodel.fit()
	print("results.params(diabetes):\n",results.params, "\nresults.summary(diabetes):\n",results.summary())

def StatsModels_linear_regression_for_LOS_WLS_for_diabetes_data():
        #Load Diabetes
        Diabetes = datasets.load_diabetes()
        X,y = Diabetes.data[:,5].reshape(-1,1),Diabetes.target.reshape(-1,1)
        testmodel = sm.OLS(y,X)
        results = testmodel.fit()
        print("OLS:\n",results.params, "\nOLS:\n",results.summary())
        Boston = datasets.load_boston()
        X,y = Diabetes.data[:,5].reshape(-1,1),Diabetes.target.reshape(-1,1)
        testmodel = sm.WLS(y,X)
        results = testmodel.fit()
        print("WLS:\n",results.params, "\nWLS:\n",results.summary())
        Boston = datasets.load_boston()
        X,y = Diabetes.data[:,5].reshape(-1,1),Diabetes.target.reshape(-1,1)
        testmodel = sm.GLS(y,X)
        results = testmodel.fit()
        print("GLS:\n",results.params, "\nGLS:\n",results.summary())
    
def StatsModels_linear_regression_for_LOS_WLS_for_boston_data():
        #Load Boston
        Boston = datasets.load_boston()
        X,y = Boston.data[:,5].reshape(-1,1),Boston.target.reshape(-1,1)
        testmodel = sm.OLS(y,X)
        results = testmodel.fit()
        print("OLS:\n",results.params, "\nOLS:\n",results.summary())
        Boston = datasets.load_boston()
        X,y = Boston.data[:,5].reshape(-1,1),Boston.target.reshape(-1,1)
        testmodel = sm.WLS(y,X)
        results = testmodel.fit()
        print("WLS:\n",results.params, "\nWLS:\n",results.summary())
        Boston = datasets.load_boston()
        X,y = Boston.data[:,5].reshape(-1,1),Boston.target.reshape(-1,1)
        testmodel = sm.GLS(y,X)
        results = testmodel.fit()
        print("GLS:\n",results.params, "\nGLS:\n",results.summary())
    
##############################################
#                   Ridge
##############################################
def Ridge_for_make_regression():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = Ridge().fit(X,y)
	print('lr.coef_: {} '.format(clf.coef_[:]))
	print('reg.intercept_: {}'.format(clf.intercept_))
	print('训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression Ridge()回归线（无噪音）"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(Ridge(),X,y,title)
	myutil.show_pic(title)

def Ridge_for_make_regression_add_noise():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = Ridge().fit(X,y)
	print('lr.coef_: {} '.format(clf.coef_[:]))
	print('reg.intercept_: {}'.format(clf.intercept_))
	print('训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression Ridge()回归线（有噪音）"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(Ridge(),X,y,title)
	myutil.show_pic(title)

#对岭回归进行分析糖尿病数据
def Ridge_for_for_diabetes():
	myutil = util()
	X,y = datasets.load_diabetes().data,datasets.load_diabetes().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	lr = LinearRegression().fit(X_train,y_train)
	print('线性回归,糖尿病数据训练集得分: {:.2%}'.format(lr.score(X_train,y_train)))
	print('线性回归,糖尿病数据测试集得分: {:.2%}'.format(lr.score(X_test,y_test)))
	title = "线性回归 糖尿病数据"
	myutil.plot_learning_curve(LinearRegression(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	ridge = Ridge().fit(X_train,y_train)
	print('alpha=1,糖尿病数据训练集得分: {:.2%}'.format(ridge.score(X_train,y_train)))
	print('alpha=1,糖尿病数据测试集得分: {:.2%}'.format(ridge.score(X_test,y_test)))
	title = "Ridge 糖尿病数据 alpha=1"
	myutil.plot_learning_curve(Ridge(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	ridge10 = Ridge(alpha=10).fit(X_train,y_train)
	print('alpha=10,糖尿病数据训练集得分: {:.2%}'.format(ridge10.score(X_train,y_train)))
	print('alpha=10,糖尿病数据测试集得分: {:.2%}'.format(ridge10.score(X_test,y_test)))
	title = "Ridge 糖尿病数据 alpha=10"
	myutil.plot_learning_curve(Ridge(alpha=10),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
	print('alpha=0.1,糖尿病数据训练集得分: {:.2%}'.format(ridge01.score(X_train,y_train)))
	print('alpha=0.1,糖尿病数据测试集得分: {:.2%}'.format(ridge01.score(X_test,y_test)))
	title = "Ridge 糖尿病数据 alpha=0.1"
	myutil.plot_learning_curve(Ridge(alpha=0.1),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	title = "Ridge 糖尿病数据 数据分布比较"
	plt.plot(ridge.coef_,'s',label='岭回归 alpha=1')
	plt.plot(ridge10.coef_,'^',label='岭回归 alpha=10')
	plt.plot(ridge01.coef_,'v',label='岭回归 alpha=0.1')
	plt.plot(lr.coef_,'o',label='线性回归 Regression')
	plt.xlabel(u'系数指数')
	plt.ylabel(u'系数大小')
	plt.hlines(0,0,len(lr.coef_))
	myutil.show_pic(title)

#对岭回归进行分析波士顿房价数据
def Ridge_for_for_boston():
	myutil = util()
	X,y = datasets.load_boston().data,datasets.load_boston().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	lr = LinearRegression().fit(X_train,y_train)
	print('线性回归,波士顿房价数据训练集得分: {:.2%}'.format(lr.score(X_train,y_train)))
	print('线性回归,波士顿房价数据测试集得分: {:.2%}'.format(lr.score(X_test,y_test)))
	title = "线性回归 波士顿房价病数据"
	myutil.plot_learning_curve(LinearRegression(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	ridge = Ridge().fit(X_train,y_train)
	print('alpha=1,波士顿房价数据训练集得分: {:.2%}'.format(ridge.score(X_train,y_train)))
	print('alpha=1,波士顿房价数据测试集得分: {:.2%}'.format(ridge.score(X_test,y_test)))
	title = "Ridge 波士顿房价数据 alpha=1"
	myutil.plot_learning_curve(Ridge(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	ridge10 = Ridge(alpha=10).fit(X_train,y_train)
	print('alpha=10,波士顿房价数据训练集得分: {:.2%}'.format(ridge10.score(X_train,y_train)))
	print('alpha=10,波士顿房价数据测试集得分: {:.2%}'.format(ridge10.score(X_test,y_test)))
	title = "Ridge 波士顿房价数据 alpha=10"
	myutil.plot_learning_curve(Ridge(alpha=10),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
	print('alpha=0.1,波士顿房价数据训练集得分: {:.2%}'.format(ridge01.score(X_train,y_train)))
	print('alpha=0.1,波士顿房价数据测试集得分: {:.2%}'.format(ridge01.score(X_test,y_test)))
	title = "Ridge 波士顿房价数据 alpha=0.1"
	myutil.plot_learning_curve(Ridge(alpha=0.1),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	title = "Ridge 波士顿房价数据 数据分布比较"
	plt.plot(ridge.coef_,'s',label='岭回归 alpha=1')
	plt.plot(ridge10.coef_,'^',label='岭回归 alpha=10')
	plt.plot(ridge01.coef_,'v',label='岭回归 alpha=0.1')
	plt.plot(lr.coef_,'o',label='线性回归 Regression')
	plt.xlabel(u'系数指数')
	plt.ylabel(u'系数大小')
	plt.hlines(0,0,len(lr.coef_))
	myutil.show_pic(title)

##############################################
#                   Lasso
##############################################	
def Lasso_for_make_regression():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = Lasso().fit(X,y)
	print('lr.coef_: {} '.format(clf.coef_[:]))
	print('reg.intercept_: {}'.format(clf.intercept_))
	print('训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression Lasso()回归线（无噪音）"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(Lasso(),X,y,title)
	myutil.show_pic(title)

def Lasso_for_make_regression_add_noise():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = Lasso().fit(X,y)
	print('lr.coef_: {} '.format(clf.coef_[:]))
	print('reg.intercept_: {}'.format(clf.intercept_))
	print('训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression Lasso()回归线（有噪音）"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(Lasso(),X,y,title)
	myutil.show_pic(title)

#对套索回归进行分析糖尿病数据
def Lasso_for_for_diabetes():
	myutil = util()
	X,y = datasets.load_diabetes().data,datasets.load_diabetes().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	lr = LinearRegression().fit(X_train,y_train)
	print('线性回归,糖尿病数据训练集得分: {:.2%}'.format(lr.score(X_train,y_train)))
	print('线性回归,糖尿病数据测试集得分: {:.2%}'.format(lr.score(X_test,y_test)))
	title = "线性回归 糖尿病数据"
	myutil.plot_learning_curve(LinearRegression(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	lasso = Lasso().fit(X_train,y_train)
	print('alpha=1,糖尿病数据训练集得分: {:.2%}'.format(lasso.score(X_train,y_train)))
	print('alpha=1,糖尿病数据测试集得分: {:.2%}'.format(lasso.score(X_test,y_test)))
	print('alpha=1,糖尿病数据套索回归特征数: {}'.format(np.sum(lasso.coef_!=0)))
	title = "Lasso 糖尿病数据 alpha=1"
	myutil.plot_learning_curve(Lasso(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	lasso10 = Lasso(alpha=10).fit(X_train,y_train)
	print('alpha=10,糖尿病数据训练集得分: {:.2%}'.format(lasso10.score(X_train,y_train)))
	print('alpha=10,糖尿病数据测试集得分: {:.2%}'.format(lasso10.score(X_test,y_test)))
	print('alpha=10,糖尿病数据套索回归特征数: {}'.format(np.sum(lasso10.coef_!=0)))
	title = "Lasso 糖尿病数据 alpha=10"
	myutil.plot_learning_curve(Lasso(alpha=10),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	lasso01 = Lasso(alpha=0.1).fit(X_train,y_train)
	print('alpha=0.1,糖尿病数据训练集得分: {:.2%}'.format(lasso01.score(X_train,y_train)))
	print('alpha=0.1,糖尿病数据测试集得分: {:.2%}'.format(lasso01.score(X_test,y_test)))
	print('alpha=0.1,糖尿病数据套索回归特征数: {}'.format(np.sum(lasso01.coef_!=0)))
	title = "Lasso 糖尿病数据 alpha= 0.1"
	myutil.plot_learning_curve(Lasso(alpha=0.1),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	title = "Lasso 糖尿病数据 数据分布比较"
	plt.plot(lasso.coef_,'s',label='套索回归 alpha=1')
	plt.plot(lasso10.coef_,'^',label='套索回归 alpha=10')
	plt.plot(lasso01.coef_,'v',label='套索回归 alpha=0.1')
	plt.plot(lr.coef_,'o',label='线性回归 Regression')
	plt.xlabel(u'系数指数')
	plt.ylabel(u'系数大小')
	plt.hlines(0,0,len(lr.coef_))
	myutil.show_pic(title)

#对套索回归进行分析波士顿房价数据
def Lasso_for_for_boston():
	myutil = util()
	X,y = datasets.load_boston().data,datasets.load_boston().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	lr = LinearRegression().fit(X_train,y_train)
	print('线性回归,波士顿房价数据训练集得分: {:.2%}'.format(lr.score(X_train,y_train)))
	print('线性回归,波士顿房价数据测试集得分: {:.2%}'.format(lr.score(X_test,y_test)))
	title = "线性回归 波士顿房价病数据"
	myutil.plot_learning_curve(LinearRegression(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	lasso = Lasso().fit(X_train,y_train)
	print('alpha=1,波士顿房价数据训练集得分: {:.2%}'.format(lasso.score(X_train,y_train)))
	print('alpha=1,波士顿房价数据测试集得分: {:.2%}'.format(lasso.score(X_test,y_test)))
	print('alpha=1,波士顿房价数据回归特征数: {}'.format(np.sum(lasso.coef_!=0)))
	title = "Lasso 波士顿房价数据 "
	myutil.plot_learning_curve(Lasso(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	lasso10 = Lasso(alpha=10).fit(X_train,y_train)
	print('alpha=10,波士顿房价数据训练集得分: {:.2%}'.format(lasso10.score(X_train,y_train)))
	print('alpha=10,波士顿房价数据测试集得分: {:.2%}'.format(lasso10.score(X_test,y_test)))
	print('alpha=10,波士顿房价数据回归特征数: {}'.format(np.sum(lasso10.coef_!=0)))
	title = "Lasso 波士顿房价数据 alpha=10"
	myutil.plot_learning_curve(Lasso(alpha=10),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	lasso01 = Lasso(alpha=0.1).fit(X_train,y_train)
	print('alpha=0.1,波士顿房价数据训练集得分: {:.2%}'.format(lasso01.score(X_train,y_train)))
	print('alpha=0.1,波士顿房价数据测试集得分: {:.2%}'.format(lasso01.score(X_test,y_test)))
	print('alpha=0.1,波士顿房价数据回归特征数: {}'.format(np.sum(lasso01.coef_!=0)))
	title = "Lasso 糖尿病数据 alpha= 0.1"
	myutil.plot_learning_curve(Lasso(alpha=0.1),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	title = "Lasso 波士顿房价数据 数据分布比较"
	plt.plot(lasso01.coef_,'s',label='套索回归 alpha=1')
	plt.plot(lasso10.coef_,'^',label='套索回归 alpha=10')
	plt.plot(lasso01.coef_,'v',label='套索回归 alpha=0.1')
	plt.plot(lr.coef_,'o',label='线性回归 Regression')
	plt.xlabel(u'系数指数')
	plt.ylabel(u'系数大小')
	plt.hlines(0,0,len(lr.coef_))
	myutil.show_pic(title)

#比较岭回归与套索回归
def Ridge_VS_Lasso():
	myutil = util()
	X,y = datasets.load_diabetes().data,datasets.load_diabetes().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	lasso = Lasso(alpha=1,max_iter=100000).fit(X_train,y_train)
	plt.plot(lasso.coef_,'s',label='lasso alpha=1')
	lasso01 = Lasso(alpha=0.1,max_iter=100000).fit(X_train,y_train)
	plt.plot(lasso01.coef_,'^',label='lasso alpha=0.1')
	lasso0001 = Lasso(alpha=0.0001,max_iter=100000).fit(X_train,y_train)
	plt.plot(lasso0001.coef_,'v',label='lasso alpha=0.001')
	ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
	plt.plot(ridge01.coef_,'o',label='ridge01 alpha=0.1')
	plt.legend(ncol=2,loc=(0,1.05))
	plt.ylim(-1000,750)
	plt.legend(loc='lower right')
	title = "比较岭回归与套索回归"
	plt.xlabel(u"系数指数")
	plt.ylabel(u"系数大小")
	myutil.show_pic(title)

##############################################
#                   弹性网络
##############################################
def ElasticNet_for_make_regression():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = ElasticNet().fit(X,y)
	print('lr.coef_: {} '.format(clf.coef_[:]))
	print('reg.intercept_: {}'.format(clf.intercept_))
	print('训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression ElasticNet()回归线（无噪音）"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(ElasticNet(),X,y,title)
	myutil.show_pic(title)

def ElasticNet_for_make_regression_add_noise():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = ElasticNet().fit(X,y)
	print('lr.coef_: {} '.format(clf.coef_[:]))
	print('reg.intercept_: {}'.format(clf.intercept_))
	print('训练集得分: {:.2%}'.format(clf.score(X_train,y_train)))
	print('测试集得分: {:.2%}'.format(clf.score(X_test,y_test)))
	title = "make_regression ElasticNet()回归线（有噪音）"
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(ElasticNet(),X,y,title)
	myutil.show_pic(title)
	
#对弹性网络进行分析糖尿病数据
def ElasticNet_for_for_diabetes():
	myutil = util()
	X,y = datasets.load_diabetes().data,datasets.load_diabetes().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	#####################################################################################
	elasticnet = ElasticNet().fit(X_train,y_train)
	print('alpha=1,糖尿病数据训练集得分: {:.2%}'.format(elasticnet.score(X_train,y_train)))
	print('alpha=1,糖尿病数据测试集得分: {:.2%}'.format(elasticnet.score(X_test,y_test)))
	print('alpha=1,糖尿病数据弹性网络回归特征数: {}'.format(np.sum(elasticnet.coef_!=0)))
	title = "ElasticNet 糖尿病数据 alpha=1"
	myutil.plot_learning_curve(ElasticNet(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	elasticnet10 = ElasticNet(alpha=10).fit(X_train,y_train)
	print('alpha=10,糖尿病数据训练集得分: {:.2%}'.format(elasticnet10.score(X_train,y_train)))
	print('alpha=10,糖尿病数据测试集得分: {:.2%}'.format(elasticnet10.score(X_test,y_test)))
	print('alpha=10,糖尿病数据弹性网络回归特征数: {}'.format(np.sum(elasticnet10.coef_!=0)))
	title = "ElasticNet 糖尿病数据 alpha=10"
	myutil.plot_learning_curve(ElasticNet(alpha=10),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	elasticnet01 = ElasticNet(alpha=0.1).fit(X_train,y_train)
	print('alpha=0.1,糖尿病数据训练集得分: {:.2%}'.format(elasticnet01.score(X_train,y_train)))
	print('alpha=0.1,糖尿病数据测试集得分: {:.2%}'.format(elasticnet01.score(X_test,y_test)))
	print('alpha=0.1,糖尿病数据弹性网络回归特征数: {}'.format(np.sum(elasticnet01.coef_!=0)))
	title = "ElasticNet 糖尿病数据 alpha= 0.1"
	myutil.plot_learning_curve(ElasticNet(alpha=0.1),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	title = "Lasso 糖尿病数据 数据分布比较"
	plt.plot(elasticnet.coef_,'s',label='弹性网络 alpha=1')
	plt.plot(elasticnet10.coef_,'^',label='弹性网络 alpha=10')
	plt.plot(elasticnet01.coef_,'v',label='弹性网络 alpha=0.1')
	plt.xlabel(u'系数指数')
	plt.ylabel(u'系数大小')
	plt.legend(loc='lower right')
	title = "比较弹性网络参数"
	myutil.show_pic(title)
	#####################################################################################
	elasticnet_01 = ElasticNet(l1_ratio=0.1).fit(X_train,y_train)
	print('l1_ratio=0.1,糖尿病数据训练集得分: {:.2%}'.format(elasticnet_01.score(X_train,y_train)))
	print('l1_ratio=0.1,糖尿病数据测试集得分: {:.2%}'.format(elasticnet_01.score(X_test,y_test)))
	print('l1_ratio=0.1,糖尿病数据弹性网络回归特征数: {}'.format(np.sum(elasticnet_01.coef_!=0)))
	title = "ElasticNet 糖尿病数据 l1_ratio=0.1"
	myutil.plot_learning_curve(ElasticNet(l1_ratio=0.1),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	elasticnet05 = ElasticNet(l1_ratio=0.5).fit(X_train,y_train)
	print('l1_ratio=0.5,糖尿病数据训练集得分: {:.2%}'.format(elasticnet05.score(X_train,y_train)))
	print('l1_ratio=0.5,糖尿病数据测试集得分: {:.2%}'.format(elasticnet05.score(X_test,y_test)))
	print('l1_ratio=0.5,糖尿病数据弹性网络回归特征数: {}'.format(np.sum(elasticnet05.coef_!=0)))
	title = "ElasticNet 糖尿病数据 l1_ratio=10"
	myutil.plot_learning_curve(ElasticNet(l1_ratio=0.5),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	elasticnet09 = ElasticNet(l1_ratio=0.9).fit(X_train,y_train)
	print('l1_ratio=0.9,糖尿病数据训练集得分: {:.2%}'.format(elasticnet09.score(X_train,y_train)))
	print('l1_ratio=0.9,糖尿病数据测试集得分: {:.2%}'.format(elasticnet09.score(X_test,y_test)))
	print('l1_ratio=0.9,糖尿病数据弹性网络回归特征数: {}'.format(np.sum(elasticnet09.coef_!=0)))
	title = "ElasticNet 糖尿病数据 l1_ratio= 0.9"
	myutil.plot_learning_curve(ElasticNet(l1_ratio=0.9),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	title = "ElasticNet 糖尿病数据 数据分布比较"
	plt.plot(elasticnet_01.coef_,'s',label='弹性网络 l1_ratio=0.1')
	plt.plot(elasticnet05.coef_,'^',label='弹性网络 l1_ratio=0.5')
	plt.plot(elasticnet09.coef_,'v',label='弹性网络 l1_ratio=0.9')
	plt.xlabel(u'系数指数')
	plt.ylabel(u'系数大小')
	plt.legend(loc='lower right')
	title = "比较弹性网络参数"
	myutil.show_pic(title)

#对弹性网络进行分析波士顿房价数据
def ElasticNet_for_for_boston():
	myutil = util()
	X,y = datasets.load_boston().data,datasets.load_boston().target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	#####################################################################################
	elasticnet = ElasticNet().fit(X_train,y_train)
	print('alpha=1,波士顿房价数据训练集得分: {:.2%}'.format(elasticnet.score(X_train,y_train)))
	print('alpha=1,波士顿房价数据测试集得分: {:.2%}'.format(elasticnet.score(X_test,y_test)))
	print('alpha=1,波士顿房价数据弹性网络回归特征数: {}'.format(np.sum(elasticnet.coef_!=0)))
	title = "ElasticNet 波士顿房价数据 alpha=1"
	myutil.plot_learning_curve(ElasticNet(),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	elasticnet10 = ElasticNet(alpha=10).fit(X_train,y_train)
	print('alpha=10,波士顿房价数据训练集得分: {:.2%}'.format(elasticnet10.score(X_train,y_train)))
	print('alpha=10,波士顿房价数据测试集得分: {:.2%}'.format(elasticnet10.score(X_test,y_test)))
	print('alpha=10,波士顿房价数据弹性网络回归特征数: {}'.format(np.sum(elasticnet10.coef_!=0)))
	title = "ElasticNet 波士顿房价数据 alpha=10"
	myutil.plot_learning_curve(ElasticNet(alpha=10),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	elasticnet01 = ElasticNet(alpha=0.1).fit(X_train,y_train)
	print('alpha=0.1,波士顿房价数据训练集得分: {:.2%}'.format(elasticnet01.score(X_train,y_train)))
	print('alpha=0.1,波士顿房价数据测试集得分: {:.2%}'.format(elasticnet01.score(X_test,y_test)))
	print('alpha=0.1,波士顿房价数据弹性网络回归特征数: {}'.format(np.sum(elasticnet01.coef_!=0)))
	title = "ElasticNet 波士顿房价数据 alpha= 0.1"
	myutil.plot_learning_curve(ElasticNet(alpha=0.1),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	title = "Lasso 波士顿房价数据 数据分布比较"
	plt.plot(elasticnet.coef_,'s',label='弹性网络 alpha=1')
	plt.plot(elasticnet10.coef_,'^',label='弹性网络 alpha=10')
	plt.plot(elasticnet01.coef_,'v',label='弹性网络 alpha=0.1')
	plt.xlabel(u'系数指数')
	plt.ylabel(u'系数大小')
	plt.legend(loc='lower right')
	title = "比较弹性网络参数"
	myutil.show_pic(title)
	#####################################################################################
	elasticnet_01 = ElasticNet(l1_ratio=0.1).fit(X_train,y_train)
	print('l1_ratio=0.1,波士顿房价数据训练集得分: {:.2%}'.format(elasticnet_01.score(X_train,y_train)))
	print('l1_ratio=0.1,波士顿房价数据测试集得分: {:.2%}'.format(elasticnet_01.score(X_test,y_test)))
	print('l1_ratio=0.1,波士顿房价数据弹性网络回归特征数: {}'.format(np.sum(elasticnet_01.coef_!=0)))
	title = "ElasticNet 波士顿房价数据 l1_ratio=0.1"
	myutil.plot_learning_curve(ElasticNet(l1_ratio=0.1),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	elasticnet05 = ElasticNet(l1_ratio=0.5).fit(X_train,y_train)
	print('l1_ratio=0.5,波士顿房价数据训练集得分: {:.2%}'.format(elasticnet05.score(X_train,y_train)))
	print('l1_ratio=0.5,波士顿房价数据测试集得分: {:.2%}'.format(elasticnet05.score(X_test,y_test)))
	print('l1_ratio=0.5,波士顿房价数据弹性网络回归特征数: {}'.format(np.sum(elasticnet05.coef_!=0)))
	title = "ElasticNet 波士顿房价数据 l1_ratio=0.5"
	myutil.plot_learning_curve(ElasticNet(l1_ratio=0.5),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	elasticnet09 = ElasticNet(l1_ratio=0.9).fit(X_train,y_train)
	print('l1_ratio=0.9,波士顿房价数据训练集得分: {:.2%}'.format(elasticnet09.score(X_train,y_train)))
	print('l1_ratio=0.9,波士顿房价数据测试集得分: {:.2%}'.format(elasticnet09.score(X_test,y_test)))
	print('l1_ratio=0.9,波士顿房价数据弹性网络回归特征数: {}'.format(np.sum(elasticnet09.coef_!=0)))
	title = "ElasticNet 波士顿房价数据 l1_ratio= 0.9"
	myutil.plot_learning_curve(ElasticNet(l1_ratio=0.9),X,y,title)
	myutil.show_pic(title)
	#####################################################################################
	title = u"ElasticNet 波士顿房价数据 数据分布比较"
	plt.plot(elasticnet_01.coef_,'s',label='弹性网络 l1_ratio=0.1')
	plt.plot(elasticnet05.coef_,'^',label='弹性网络 l1_ratio=0.5')
	plt.plot(elasticnet09.coef_,'v',label='弹性网络 l1_ratio=0.9')
	plt.xlabel(u'系数指数')
	plt.ylabel(u'系数大小')
	plt.legend(loc='lower right')
	title = u"比较弹性网络参数"
	myutil.show_pic(title)
				 
############################################################################################
#                   Summay
############################################################################################
def Analysis_boston_data():
	myutil = util()
	ax = plt.gca()
	boston = datasets.load_boston()
	X,y = boston.data,boston.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	#将特征数字中的最大和最小值一散点图形式画出来
	plt.plot(X.min(axis=0),'v',label='最小')
	plt.plot(X.max(axis=0),'^',label='最大')
	#纵坐标为对数形式
	plt.yscale('log')
	#设置图注位置最佳
	plt.legend(loc='best')
	ax.set_xlabel(u'特征')
	ax.set_ylabel(u'特征量')
	title = u"波士顿房价数据分析"
	myutil.show_pic(title)
	############################################################################################
	#对训练集和测试集数据进行预处理
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaler = scaler.transform(X_train)
	X_test_scaler = scaler.transform(X_test)
	plt.plot(X_train_scaler.min(axis=0),'v',label=u'训练集最小')
	plt.plot(X_train_scaler.max(axis=0),'^',label=u'训练集最大')
	plt.plot(X_test_scaler.min(axis=0),'v',label=u'测试集最小')
	plt.plot(X_test_scaler.max(axis=0),'^',label=u'测试集最大')
	plt.legend(loc='best')
	ax.set_xlabel(u'预处理特征')
	ax.set_ylabel(u'预处理特征量')
	title = u"预处理后的波士顿房价数据分析"
	myutil.show_pic(title)
	############################################################################################
	#用线性回归最优解处理
	clf = LinearRegression()
	clf = clf.fit(X_train_scaler,y_train)
	print('线性回归波士顿房价训练集得分: {:.2%}'.format(clf.score(X_train_scaler,y_train)))
	print('线性回归波士顿房价测试集得分: {:.2%}'.format(clf.score(X_test_scaler,y_test)))
	############################################################################################
	#用岭回归最优解处理
	clf = Ridge(alpha=0.1)
	clf = clf.fit(X_train_scaler,y_train)
	print('岭回归波士顿房价训练集得分: {:.2%}'.format(clf.score(X_train_scaler,y_train)))
	print('岭回归波士顿房价测试集得分: {:.2%}'.format(clf.score(X_test_scaler,y_test)))
	############################################################################################
	#用套索回归最优解处理
	clf = Lasso(alpha=10)
	clf = clf.fit(X_train_scaler,y_train)
	print('套索回归波士顿房价训练集得分: {:.2%}'.format(clf.score(X_train_scaler,y_train)))
	print('套索回归波士顿房价测试集得分: {:.2%}'.format(clf.score(X_test_scaler,y_test)))
	############################################################################################
	#用弹性网络回归最优解处理
	clf = ElasticNet(alpha=0.1,l1_ratio=0.5)
	clf = clf.fit(X_train_scaler,y_train)
	print('弹性网络波士顿房价训练集得分: {:.2%}'.format(clf.score(X_train_scaler,y_train)))
	print('弹性网络波士顿房价测试集得分: {:.2%}'.format(clf.score(X_test_scaler,y_test)))

def Analysis_diabetes_data():
	myutil = util()
	ax = plt.gca()
	diabetes = datasets.load_diabetes()
	X,y = diabetes.data,diabetes.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	#将特征数字中的最大和最小值一散点图形式画出来
	plt.plot(X.min(axis=0),'v',label='最小')
	plt.plot(X.max(axis=0),'^',label='最大')
	#纵坐标为对数形式
	plt.yscale('log')
	#设置图注位置最佳
	plt.legend(loc='best')
	ax.set_xlabel(u'特征')
	ax.set_ylabel(u'特征量')
	title = u"糖尿病数据分析"
	myutil.show_pic(title)
	############################################################################################
	#对训练集和测试集数据进行预处理
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaler = scaler.transform(X_train)
	X_test_scaler = scaler.transform(X_test)
	plt.plot(X_train_scaler.min(axis=0),'v',label=u'训练集最小')
	plt.plot(X_train_scaler.max(axis=0),'^',label=u'训练集最大')
	plt.plot(X_test_scaler.min(axis=0),'v',label=u'测试集最小')
	plt.plot(X_test_scaler.max(axis=0),'^',label=u'测试集最大')
	plt.legend(loc='best')
	ax.set_xlabel(u'预处理特征')
	ax.set_ylabel(u'预处理特征量')
	title = u"预处理后的糖尿病数据分析"
	myutil.show_pic(title)
	############################################################################################
	#用线性回归最优解处理
	clf = LinearRegression()
	clf = clf.fit(X_train_scaler,y_train)
	print('线性回归糖尿病训练集得分: {:.2%}'.format(clf.score(X_train_scaler,y_train)))
	print('线性回归糖尿病测试集得分: {:.2%}'.format(clf.score(X_test_scaler,y_test)))
	############################################################################################
	#用岭回归最优解处理
	clf = Ridge(alpha=0.1)
	clf = clf.fit(X_train_scaler,y_train)
	print('岭回归糖尿病训练集得分: {:.2%}'.format(clf.score(X_train_scaler,y_train)))
	print('岭回归糖尿病测试集得分: {:.2%}'.format(clf.score(X_test_scaler,y_test)))
	############################################################################################
	#用套索回归最优解处理
	clf = Lasso(alpha=0.1)
	clf = clf.fit(X_train_scaler,y_train)
	print('套索回归糖尿病训练集得分: {:.2%}'.format(clf.score(X_train_scaler,y_train)))
	print('套索回归糖尿病测试集得分: {:.2%}'.format(clf.score(X_test_scaler,y_test)))
	############################################################################################
	#用弹性网络回归最优解处理
	clf = ElasticNet(alpha=0.1,l1_ratio=0.5)
	clf = clf.fit(X_train_scaler,y_train)
	print('弹性网络糖尿病训练集得分: {:.2%}'.format(clf.score(X_train_scaler,y_train)))
	print('弹性网络糖尿病测试集得分: {:.2%}'.format(clf.score(X_test_scaler,y_test)))

if __name__=="__main__":
	#Line_base_by_k_b()
	#Line_base_by_two_point()
	#Line_base_by_three_point()
	#Line_base_by_multiple_point()
        ##########################################################
	#LinearRegression_for_make_regression()
	#LinearRegression_for_make_regression_add_noise()
	#LinearRegression_for_diabetes()
	#LinearRegression_for_boston()
        ##########################################################
	#StatsModels_linear_regression()
	#StatsModels_linear_regression_for_boston_data()
        #StatsModels_linear_regression_for_LOS_WLS_for_diabetes_data()
        #StatsModels_linear_regression_for_LOS_WLS_for_boston_data()
        ##########################################################
	#LogisticRegression_for_make_blobs()
	LogisticRegression_for_load_breast_cancer()
	LogisticRegression_for_load_iris()
	LogisticRegression_for_load_wine()
        ##########################################################
	#Ridge_for_make_regression()
	#Ridge_for_make_regression_add_noise()
	#Ridge_for_for_diabetes()
	#Ridge_for_for_boston()
        ##########################################################
	#Lasso_for_make_regression()
	#Lasso_for_make_regression_add_noise()
	#Lasso_for_for_diabetes()
	#Lasso_for_for_boston()
	#Ridge_VS_Lasso()
        ##########################################################
        #ElasticNet_for_make_regression()
        #ElasticNet_for_make_regression_add_noise()
        #ElasticNet_for_for_diabetes()
        #ElasticNet_for_for_boston()
        ##########################################################
	#SVM()
	#sklearn_datasets_data()
	#linear_for_all_data_and_model()
	#using_mglearn()
	#useing_sklearn_datasets_for_ElasticNet()
	#my_ElasticNet()
        #Analysis_boston_data()
        #Analysis_diabetes_data()
