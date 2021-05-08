# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures,StandardScaler
#OneHotEncoder独热编码，同pd.get_dummies，但仅能用于整型变量
from sklearn.feature_selection import SelectPercentile,SelectFromModel,RFE
from sklearn.linear_model import LinearRegression
import warnings

def base_for_dummies():
	colour = pd.DataFrame({'数据特征':[1,2,3,4,5,6,7],'颜色类型':['赤','橙','黄','绿','青','蓝','紫']})
	print(colour)
	colour_dum = pd.get_dummies(colour)
	print(colour_dum)
	colour['数据特征'] = colour['数据特征'].astype(str)
	colour_dum = pd.get_dummies(colour,columns=['数据特征'])
	print(colour_dum)

def different_algorithm_for_same_data():
	#生成随机数列
	rnd = np.random.RandomState(56)
	x = rnd.uniform(-5,5,size=50)
	#向数据添加噪音
	y_no_noise = (np.cos(6*x)+x)
	X = x.reshape(-1,1)
	y = (y_no_noise + rnd.normal(size=len(x)))/2
	plt.plot(X,y,'o',c='g')
	plt.show()
	#######################################################
	line = np.linspace(-5,5,1000,endpoint=False).reshape(-1,1)
	mlpr = MLPRegressor().fit(X,y)
	knr = KNeighborsRegressor().fit(X,y)
	plt.plot(line,mlpr.predict(line),label='MLP')
	plt.plot(line,knr.predict(line),label='KNN')
	plt.plot(X,y,'o',c='g')
	plt.legend(loc='best')
	plt.show()
	#######################################################'''
	#设置11个箱子
	bins = np.linspace(-5,5,11)
	#将数据装箱
	target_bin = np.digitize(X, bins=bins)
	print("装箱数据范围：\n{}".format(bins))
	print("前10个数据点的特征值：\n{}".format(X[:10]))
	print("前10个数据点所在的箱子：\n{}".format(target_bin[:10]))
	#######################################################
	onehot = OneHotEncoder(sparse = False)
	onehot.fit(target_bin)
	#将独热编码转为数据
	X_in_bin = onehot.transform(target_bin)
	print("装箱后的数据形态：\n{}".format(X_in_bin.shape))
	print("装箱后的前10个数据点：\n{}".format(X_in_bin[:10]))
	#######################################################
	#使用独热编码进行数据表达
	onehot_line = onehot.transform(np.digitize(line,bins=bins))
	onehot_mlpr = MLPRegressor().fit(X_in_bin,y)
	onehot_knr = KNeighborsRegressor().fit(X_in_bin,y)
	'''plt.plot(line,onehot_mlpr.predict(onehot_line),label='New MLP')
	plt.plot(line,onehot_knr.predict(onehot_line),label='New KNN')
	plt.plot(X,y,'o',c='g')
	plt.legend(loc='best')
	plt.show()'''
	#######################################################
	array_1 = [0,1,2,3,4]
	array_2 = [5,6,7,8,9]
	array3 = np.hstack((array_1,array_2))
	print("将数组2添加到数据1后面去得到:\n{}".format(array3))
	#######################################################
	#将原始数据和装箱数据进行堆叠
	X_stack = np.hstack([X,X_in_bin])
	print("X.shape:\n",X.shape)
	print("X_in_bin.shape:\n",X_in_bin.shape)
	print("X_stack.shape:\n",X_stack.shape)
	#######################################################
	#将数据进行堆叠
	line_stack = np.hstack([line,onehot_line])
	mlpr_interact = MLPRegressor().fit(X_stack,y)
	'''plt.plot(line,mlpr_interact.predict(line_stack),label='MLP for interaction')
	plt.ylim(-4,4)
	for vline in bins:
		plt.plot([vline,vline],[-5,5],":",c='k')
	plt.legend(loc='lower right')
	plt.plot(X,y,"o",c='g')
	plt.show()'''
	#######################################################
	#使用新的叠堆方式处理数据
	X_multi = np.hstack([X_in_bin,X*X_in_bin])
	print("X_multi.shape:\n",X_multi.shape)
	print("X_multi[0]:\n",X_multi[0])
	#######################################################
	#重新训练
	mlpr_multi = MLPRegressor().fit(X_multi,y)
	line_multi = np.hstack([onehot_line,line*onehot_line])
	plt.plot(line,mlpr_multi.predict(line_multi),label='MLP for Regressor')
	plt.ylim(-4,4)
	for vline in bins:
		plt.plot([vline,vline],[-5,5],":",c='k')
	plt.legend(loc='lower right')
	plt.plot(X,y,"o",c='g')
	plt.show()

def polynomial():
	rnd = np.random.RandomState(56)
	x = rnd.uniform(-5,5,size=50)
	X = x.reshape(-1,1)
	y_no_noise = (np.cos(6*x)+x)
	y = (y_no_noise + rnd.normal(size=len(x)))/2
	poly = PolynomialFeatures(degree=20,include_bias=False)
	X_poly = poly.fit_transform(X)
	print(X_poly.shape)
	#######################################################
	print("原始数据第一个样本：\n{}".format(X[0]))
	print("多项式处理后第一个样本：\n{}".format(X_poly[0]))
	print("PolynomialFeatures对原始数据的处理：\n{}".format(poly.get_feature_names()))
	#######################################################
	line = np.linspace(-5,5,1000,endpoint=False).reshape(-1,1)
	LNR_poly = LinearRegression().fit(X_poly,y)
	line_poly = poly.transform(line)
	plt.plot(line,LNR_poly.predict(line_poly),label='Line Regressor')
	plt.xlim(np.min(X)-0.5,np.max(X)+0.5)
	plt.ylim(np.min(y)-0.5,np.max(y)+0.5)
	plt.plot(X,y,"o",c='g')
	plt.legend(loc='lower right')
	plt.show()

def dealdata(mydata):
	if str(mydata)[-1]=='%':
		try:
			return (float(str(mydata)[:-1])/100)
		except:
			return mydata
	elif str(mydata)[-2:]=='万亿':
		try:
			return float(str(mydata)[:-2])*1000000000000
		except:
			return mydata
	elif str(mydata)[-1]=='万':
		try:
			return float(str(mydata)[:-1])*10000
		except:
			return mydata
	elif str(mydata)[-1]=='亿':
		try:
			return float(str(mydata)[:-1])*100000000
		except:
			return mydata
	elif str(mydata)=='-':
		try:
			return 0
		except:
			return mydata
	else:
		return mydata      

	
def dealcsv():
	stock = pd.read_csv('000001.csv',encoding='GBK')
	new_stock = stock.applymap(dealdata)
	print(new_stock.head())
	new_stock.to_csv('stock1.csv',encoding='GBK')

def stock():
	stock = pd.read_csv('stock.csv',encoding='GBK')
	print(stock.head())
	#######################################################
	#设置目标
	y = stock['涨跌幅']
	print(y.shape)
	print(y[0])
	#######################################################
	#提取特征值
	features = stock.loc[:,'价格':'流通市值']
	X = features.values
	print(X.shape)
	print(X[1])
	#######################################################
	mlp = MLPRegressor(random_state=53,hidden_layer_sizes=[100,100,100],alpha=0.1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=62)
	#预处理
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	mlp.fit(X_train_scaled,y_train)
	print('训练集得分:{:.2%}'.format(mlp.score(X_train_scaled,y_train)))
	print('测试集得分:{:.2%}'.format(mlp.score(X_test_scaled,y_test)))
	#######################################################
	wanted = stock.loc[:,'名称']
	print(wanted[y>=-0.05])
	#######################################################
	select = SelectPercentile(percentile=50)
	select.fit(X_train_scaled,y_train)
	X_train_selected = select.transform(X_train_scaled)
	print('经过缩放后的形态:{}'.format(X_train_scaled.shape))
	print('特征选择后的形态:{}'.format(X_train_selected.shape))
	#######################################################
	mask = select.get_support()
	print(mask)
	#用图像表示单一变量法特征选择结果
	plt.matshow(mask.reshape(1,-1),cmap=plt.cm.cool)
	plt.xlabel(u"特征选择")
	plt.rcParams['font.sans-serif']=['SimHei']
	plt.rcParams['axes.unicode_minus']=False
	plt.show()
	#######################################################
	#使用特征选择后数据集训练神经网络
	X_test_selected = select.transform(X_test_scaled)
	mlp_select = MLPRegressor(random_state=53,hidden_layer_sizes=[100,100,100],alpha=0.1)
	mlp_select.fit(X_train_selected,y_train)
	print('单一变量法特征选择后训练集得分:{:.2%}'.format(mlp_select.score(X_train_selected,y_train)))
	print('单一变量法特征选择后测试集得分:{:.2%}'.format(mlp_select.score(X_test_selected,y_test)))
	#######################################################
	
def selectFromModel():
	stock = pd.read_csv('stock.csv',encoding='GBK')
	y = stock['涨跌幅']
	features = stock.loc[:,'价格':'流通市值']
	X = features.values
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=62)
	#预处理
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	sfm = SelectFromModel(estimator=RandomForestRegressor(n_estimators=100,random_state=38),threshold='median')
	sfm.fit(X_train_scaled,y_train)
	X_train_sfm = sfm.transform(X_train_scaled)                    
	print('经过随机森林模型特征选择后的的数据形态:{}'.format(X_train_sfm.shape))
	mask = sfm.get_support()
	print(mask)
	#######################################################
	#用图像表示模型特征选择结果
	plt.matshow(mask.reshape(1,-1),cmap=plt.cm.cool)
	plt.xlabel(u"特征选择")
	plt.rcParams['font.sans-serif']=['SimHei']
	plt.rcParams['axes.unicode_minus']=False
	plt.show()
	#######################################################
	#使用随机森林模型特征选择后数据集训练随机森林
	X_test_sfm = sfm.transform(X_test_scaled)
	mlp_sfm = MLPRegressor(random_state=53,hidden_layer_sizes=[100,100,100],alpha=0.1)
	mlp_sfm.fit(X_train_sfm,y_train)
	print('经过随机森林模型特征选择后训练集得分:{:.2%}'.format(mlp_sfm.score(X_train_sfm,y_train)))
	print('经过随机森林模型特征选择后测试集得分:{:.2%}'.format(mlp_sfm.score(X_test_sfm,y_test)))

def elimination():
	stock = pd.read_csv('stock.csv',encoding='GBK')
	y = stock['涨跌幅']
	features = stock.loc[:,'价格':'流通市值']
	X = features.values
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=62)
	#预处理
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	rfe = RFE(RandomForestRegressor(n_estimators=100,random_state=38),n_features_to_select=8)
	rfe.fit(X_train_scaled,y_train)
	X_train_rfe = rfe.transform(X_train_scaled)
	print('经过随机森林模型进行迭代特征选择后的的数据形态:{}'.format(X_train_rfe.shape))
	mask = rfe.get_support()
	print(mask)
	#######################################################
	#用图像表示特征选择结果
	plt.matshow(mask.reshape(1,-1),cmap=plt.cm.cool)
	plt.xlabel(u"特征选择")
	plt.rcParams['font.sans-serif']=['SimHei']
	plt.rcParams['axes.unicode_minus']=False
	plt.show()
	#######################################################
	#使用随机森林迭代特征选择后数据集训练随机森林
	X_test_rfe = rfe.transform(X_test_scaled)
	mlp_rfe = MLPRegressor(random_state=53,hidden_layer_sizes=[100,100,100],alpha=0.1)
	mlp_rfe.fit(X_train_rfe,y_train)
	print('经过随机森林迭代特征选择后训练集得分:{:.2%}'.format(mlp_rfe.score(X_train_rfe,y_train)))
	print('经过随机森林迭代特征选择后测试集得分:{:.2%}'.format(mlp_rfe.score(X_test_rfe,y_test)))

if __name__=="__main__":
	#base_for_dummies()
	#different_algorithm_for_same_data()
	#polynomial()
	#stock()
	#dealcsv()
	#selectFromModel()
	elimination()