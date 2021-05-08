# coding:utf-8

# 导入NumPy库 
import numpy as np
# 导入画图工具
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import tree,datasets
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.tree import export_graphviz#pip3 install graphviz
# Graphviz 是一款由 AT&T Research 和 Lucent Bell 实验室开源的可视化图形工具
import graphviz
import mglearn
from Util import util

##############################################################################################
# DecisionTreeClassifier
##############################################################################################
def iris_of_decision_tree():
	myutil = util()
	iris = datasets.load_iris()
	# 仅选前两个特征
	X = iris.data[:,:2]
	y = iris.target
	X_train,X_test,y_train,y_test = train_test_split(X, y)
	for max_depth in [1,3,5,7]:
		clf = DecisionTreeClassifier(max_depth=max_depth)
		clf.fit(X_train,y_train)
		title=u"鸢尾花数据测试集(max_depth="+str(max_depth)+")"
		myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
		myutil.draw_scatter(X,y,clf,title)
		myutil.plot_learning_curve(DecisionTreeClassifier(max_depth=max_depth),X,y,title)
		myutil.show_pic(title)

def wine_of_decision_tree():
	myutil = util()
	wine = datasets.load_wine()
	# 仅选前两个特征
	X = wine.data[:,:2]
	y = wine.target
	X_train,X_test,y_train,y_test = train_test_split(X, y)
	for max_depth in [1,3,5,7]:
		clf = DecisionTreeClassifier(max_depth=max_depth)
		clf.fit(X_train,y_train)
		title=u"红酒数据测试集(max_depth="+str(max_depth)+")"
		myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
		myutil.draw_scatter(X,y,clf,title)
		myutil.plot_learning_curve(DecisionTreeClassifier(max_depth=max_depth),X,y,title)
		myutil.show_pic(title)

def breast_cancer_of_decision_tree():
	myutil = util()
	breast_cancer = datasets.load_breast_cancer()
	# 仅选前两个特征
	X = breast_cancer.data[:,:2]
	y = breast_cancer.target
	X_train,X_test,y_train,y_test = train_test_split(X, y)
	for max_depth in [1,3,5,7]:
		clf = DecisionTreeClassifier(max_depth=max_depth)
		clf.fit(X_train,y_train)
		title=u"乳腺癌数据测试集(max_depth="+str(max_depth)+")"
		myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
		myutil.draw_scatter(X,y,clf,title)
		myutil.plot_learning_curve(DecisionTreeClassifier(max_depth=max_depth),X,y,title)
		myutil.show_pic(title)

##############################################################################################
# DecisionTreeRegressor
##############################################################################################
#加入噪音
def DecisionTreeRegressor_for_make_regression_add_noise():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = DecisionTreeRegressor().fit(X,y)
	title = "make_regression DecisionTreeRegressor()回归线（有噪音）"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(DecisionTreeRegressor(),X,y,title)
	myutil.show_pic(title)

#分析波士顿房价数据
def DecisionTreeRegressor_for_boston():
	myutil = util()
	boston = datasets.load_boston()
	X,y = boston.data,boston.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	for max_depth in [1,3,5,7]:
		clf = DecisionTreeRegressor(max_depth=max_depth)
		clf.fit(X_train,y_train)
		title=u"波士顿据测试集(max_depth="+str(max_depth)+")"
		myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
		myutil.plot_learning_curve(DecisionTreeRegressor(max_depth=max_depth),X,y,title)
		myutil.show_pic(title)

#分析糖尿病数据
def DecisionTreeRegressor_for_diabetes():
	myutil = util()
	diabetes = datasets.load_diabetes()
	X,y = diabetes.data,diabetes.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	for max_depth in [1,3,5,7]:
		clf = DecisionTreeRegressor(max_depth=max_depth)
		clf.fit(X_train,y_train)
		title=u"糖尿病据测试集(max_depth="+str(max_depth)+")"
		myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
		myutil.plot_learning_curve(DecisionTreeRegressor(max_depth=max_depth),X,y,title)
		myutil.show_pic(title)
				 
def show_tree():
	wine = datasets.load_wine()
	# 仅选前两个特征
	X = wine.data[:,:2]
	y = wine.target
	X_train,X_test,y_train,y_test = train_test_split(X, y)
	clf = DecisionTreeClassifier(max_depth=3)#为了图片不太大选择max_depth=3
	clf.fit(X_train,y_train)
	export_graphviz(clf,out_file="wine.dot",class_names=wine.target_names,feature_names=wine.feature_names[:2],impurity=False,filled=True)
	#打开dot文件
	with open("wine.dot") as f:
		dot_graph = f.read()
	graphviz.Source(dot_graph)
	
def decision_tree_pruning():
	myutil = util()
	cancer = datasets.load_breast_cancer()
	X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=42)#stratify:分层
	# 构件树，不剪枝
	tree = DecisionTreeClassifier(random_state=0)
	tree.fit(X_train,y_train)
	title = "不剪枝，训练数据集上的精度"
	myutil.print_scores(tree,X_train,y_train,X_test,y_test,title)
	print("不剪枝，树的深度:{}".format(tree.get_depth()))
	# 构件树，剪枝
	tree = DecisionTreeClassifier(max_depth=4,random_state=0)
	tree.fit(X_train,y_train)
	title = "剪枝，训练数据集上的精度"
	myutil.print_scores(tree,X_train,y_train,X_test,y_test,title)
	print("剪枝，树的深度:{}".format(tree.get_depth()))

##############################################################################################
#RandomForestClassifier
##############################################################################################
def base_of_decision_tree_forest(n_estimator,random_state,X,y,title):
	myutil = util()
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	clf = RandomForestClassifier(n_estimators=n_estimator, random_state=random_state,n_jobs=2)#n_jobs:设置为CPU个数
	# 在训练数据集上进行学习
	clf.fit(X_train, y_train)
	cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
	cmap_bold =  ListedColormap(['#FF0000','#00FF00','#0000FF'])
	#分别将样本的两个特征值创建图像的横轴和纵轴
	x_min,x_max = X_train[:,0].min()-1,X_train[:,0].max()+1
	y_min,y_max = X_train[:,1].min()-1,X_train[:,1].max()+1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
			     np.arange(y_min, y_max, .02))
	#给每个样本分配不同的颜色
	Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.pcolormesh(xx,yy,Z,cmap=cmap_light,shading='auto')

	#用散点把样本表示出来
	plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,s=20,edgecolors='k')
	plt.xlim(xx.min(),xx.max()) 
	plt.ylim(yy.min(),yy.max())
	title = title+"数据随机森林训练集得分(n_estimators:"+str(n_estimator)+",random_state:"+str(random_state)+")"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)

def tree_forest():
	myutil = util()
	title = ["鸢尾花","红酒","乳腺癌"]
	j = 0
	for datas in [datasets.load_iris(),datasets.load_wine(),datasets.load_breast_cancer()]:
		#定义图像中分区的颜色和散点的颜色
		figure,axes = plt.subplots(4,4,figsize =(100,10))
		plt.subplots_adjust(hspace=0.95)
		i = 0
		# 仅选前两个特征
		X = datas.data[:,:2]
		y = datas.target
		mytitle =title[j]
		for n_estimator in range(4,8):
			for random_state in range(2,6):
				plt.subplot(4,4,i+1)
				plt.title("n_estimator:"+str(n_estimator)+"random_state:"+str(random_state))
				plt.suptitle(u"随机森林分类")
				base_of_decision_tree_forest(n_estimator,random_state,X,y,mytitle)
				i = i + 1
		myutil.show_pic(mytitle)
		j = j+1

##############################################################################################
#RandomForestRegressor
##############################################################################################
#加入噪音
def DecisionTreeRegressor_for_make_regression_add_noise():
	myutil = util()
	X,y = make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = DecisionTreeRegressor().fit(X,y)
	title = "make_regression DecisionTreeRegressor()回归线（有噪音）"
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
	myutil.draw_line(X[:,0],y,clf,title)
	myutil.plot_learning_curve(DecisionTreeRegressor(),X,y,title)
	myutil.show_pic(title)

#分析波士顿房价数据
def DecisionTreeRegressor_for_boston():
	myutil = util()
	boston = datasets.load_boston()
	X,y = boston.data,boston.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	for max_depth in [1,3,5,7]:
		clf = DecisionTreeRegressor(max_depth=max_depth)
		clf.fit(X_train,y_train)
		title=u"波士顿据测试集(max_depth="+str(max_depth)+")"
		myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
		myutil.plot_learning_curve(DecisionTreeRegressor(max_depth=max_depth),X,y,title)
		myutil.show_pic(title)

#分析糖尿病数据
def DecisionTreeRegressor_for_diabetes():
	myutil = util()
	diabetes = datasets.load_diabetes()
	X,y = diabetes.data,diabetes.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state =8)
	for max_depth in [1,3,5,7]:
		clf = DecisionTreeRegressor(max_depth=max_depth)
		clf.fit(X_train,y_train)
		title=u"糖尿病据测试集(max_depth="+str(max_depth)+")"
		myutil.print_scores(clf,X_train,y_train,X_test,y_test,title)
		myutil.plot_learning_curve(DecisionTreeRegressor(max_depth=max_depth),X,y,title)
		myutil.show_pic(title)

##############################################################################################
# case for RandomForest
##############################################################################################
def income_forecast():
	##1-数据准备
	#1.1导入数据
	#用pandas打开csv文件
	data=pd.read_csv('adult.csv', header=None,index_col=False,
		  names=['年龄','单位性质','权重','学历','受教育时长',
			'婚姻状况','职业','家庭情况','种族','性别',
			'资产所得','资产损失','周工作时长','原籍',
			'收入'])
	#为了方便展示，我们选取其中一部分数据
	data_title = data[['年龄','单位性质','学历','性别','周工作时长','职业','收入']]
	print(data_title.head())
	#利用shape方法获取数据集的大小
	print("data_title.shape:\n",data_title.shape)
	data_title.info()
	##1-数据准备
	#1.2 数据预处理
	#用get_dummies将文本数据转化为数值
	data_dummies=pd.get_dummies(data_title)
	print("data_dummies.shape:\n",data_dummies.shape)
	#对比样本原始特征和虚拟变量特征---df.columns获取表头
	print('样本原始特征:\n',list(data_title.columns),'\n')
	print('虚拟变量特征:\n',list(data_dummies.columns))
	print(data_dummies.head())
	#1.3 选择特征
	#按位置选择---位置索引---df.iloc[[行1，行2]，[列1，列2]]---行列位置从0开始，多行多列用逗号隔开，用:表示全部(不需要[])
	#选择除了收入外的字段作为数值特征并赋值给x---df[].values
	x=data_dummies.loc[:,'年龄':'职业_ Transport-moving'].values
	#将'收入_ >50K'‘作为预测目标y
	y = data_dummies['收入_ >50K'].values
	#查看x,y数据集大小情况
	print('特征形态:{} 标签形态:{}'.format(x.shape, y.shape))
	##2-数据建模---拆分数据集/模型训练/测试
	#2.1将数据拆分为训练集和测试集---要用train_test_split模块中的train_test_split()函数，随机将75%数据化为训练集，25%数据为测试集
	#导入数据集拆分工具  
	#拆分数据集---x,y都要拆分，rain_test_split(x,y,random_state=0),random_state=0使得每次生成的伪随机数不同
	x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
	#查看拆分后的数据集大小情况
	print('x_train_shape:{}'.format(x_train.shape))
	print('x_test_shape:{}'.format(x_test.shape))
	print('y_train_shape:{}'.format(y_train.shape))
	print('y_test_shape:{}'.format(y_test.shape))
	##2、数据建模---模型训练/测试---决策树算法
	#2.2 模型训练---算法.fit(x_train,y_train)
	#使用算法
	tree = DecisionTreeClassifier(max_depth=5)#这里参数max_depth最大深度设置为5
	#算法.fit(x,y)对训练数据进行拟合
	tree.fit(x_train, y_train)
	##2、数据建模---拆分数据集/模型训练/测试---决策树算法
	#2.3 模型测试---算法.score(x_test,y_test)
	score_test=tree.score(x_test,y_test)
	score_train=tree.score(x_train,y_train)
	print('test_score:{:.2%}'.format(score_test))
	print('train_score:{:.2%}'.format(score_train))
	##3、模型应用---算法.predict(x_new)---决策树算法
	#导入要预测数据--可以输入新的数据点，也可以随便取原数据集中某一数据点，但是注意要与原数据结构相同
	x_new=[[37,40,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]]#37岁，机关工作，硕士，男，每周工作40小时，文员
	#[         '年龄', '周工作时长', '单位性质_ ?', '单位性质_ Federal-gov', '单位性质_ Local-gov', '单位性质_ Never-worked', '单位性质_ Private', '单位性质_ Self-emp-inc',
	#模型应用   37,     40,           0,             0,                       0,                     0,                        0,                   0,                        
	#单位性质_ Self-emp-not-inc', '单位性质_ State-gov', '单位性质_ Without-pay', '学历_ 10th', '学历_ 11th', '学历_ 12th', '学历_ 1st-4th', '学历_ 5th-6th', '学历_ 7th-8th',
	# 1,                           0,                      0,                      0,            0,            0,            0,               0,               0,               
	# '学历_ 9th', '学历_ Assoc-acdm', '学历_ Assoc-voc', '学历_ Bachelors', '学历_ Doctorate', '学历_ HS-grad', '学历_ Masters', '学历_ Preschool', '学历_ Prof-school', 
	# 0,           0,                  0,                  0,                0,                  0,              1,               0,                  0,                               
	#'学历_ Some-college', '性别_ Female', '性别_ Male', '职业_ ?', '职业_ Adm-clerical', '职业_ Armed-Forces', '职业_ Craft-repair', '职业_ Exec-managerial', '职业_ Farming-fishing', 
	# 0,                    1,              0,            1,         0,                    0,                    0,                    0,                       0,                       
	# '职业_ Handlers-cleaners', '职业_ Machine-op-inspct', '职业_ Other-service', '职业_ Priv-house-serv', '职业_ Prof-specialty', '职业_ Protective-serv', '职业_ Sales', '职业_ Tech-support', '职业_ Transport-moving'
	#0,                           0,                         0,                     0,                       0,                      0,                       0,             0,                    0	
	prediction=tree.predict(x_new)
	print('预测数据:{}'.format(x_new))
	print('预测结果:{}'.format(prediction))

def moons_data_for__RandomForet():
	# 生成一个用于模拟的二维数据集
	X, y = datasets.make_moons(n_samples=100, noise=0.25, random_state=3)
	# 训练集和测试集的划分
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42)
	# 初始化一个包含 5 棵决策树的随机森林分类器
	forest = RandomForestClassifier(n_estimators=5, random_state=2)
	# 在训练数据集上进行学习
	forest.fit(X_train, y_train)
	# 可视化每棵决策树的决策边界
	fig, axes = plt.subplots(2, 3, figsize=(20, 10))
	for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
		ax.set_title('Tree {}'.format(i))
		mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
		print("决策树"+str(i)+"训练集得分:{:.2%}".format(tree.score(X_train,y_train)))
		print("决策树"+str(i)+"测试集得分:{:.2%}".format(tree.score(X_test,y_test)))
	# 可视化集成分类器的决策边界
	print("随机森林训练集得分:{:.2%}".format(forest.score(X_train,y_train)))
	print("随机森林测试集得分:{:.2%}".format(forest.score(X_test,y_test)))
	mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],alpha=0.4)
	axes[-1, -1].set_title('Random Forest')
	mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
	plt.show()

if __name__=="__main__":
        # wine_of_decision_tree()
        # iris_of_decision_tree()
        # breast_cancer_of_decision_tree()
        # decision_tree_pruning()
        # DecisionTreeRegressor_for_make_regression_add_noise()
        # DecisionTreeRegressor_for_boston()
        DecisionTreeRegressor_for_diabetes()
	# show_tree()
	# tree_forest()
	# income_forecast()
	# moons_data_for__RandomForet
