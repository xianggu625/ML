# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import warnings
from PIL import Image
from Util import util

def relu_and_tanh():
	line = np.linspace(-5,5,200)
	plt.plot(line,np.tanh(line),label='tanh')#np.tanh()返回具有三角正切正弦的数组
	plt.plot(line,np.maximum(line,0),label='relu')
	plt.legend(loc='best')
	plt.xlabel('x')
	plt.ylabel('relu(x) and tanh(x)')
	plt.show()

##########################################################################################################################
#MLPClassifier
##########################################################################################################################
def My_MLPClassifier(solver,hidden_layer_sizes,activation,level,alpha,mydata,title):
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = mydata.data,mydata.target
	X1 = X[:,:2]
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0)
	mlp = MLPClassifier(solver=solver,hidden_layer_sizes=hidden_layer_sizes,activation=activation,alpha=alpha,max_iter=10000)
	mlp.fit(X_train,y_train)
	mytitle = "MLPClassifier("+title+"):solver:"+solver+",node:"+str(hidden_layer_sizes)+",activation:"+activation+",level="+str(level)+",alpha="+str(alpha)
	myutil.print_scores(mlp,X_train,y_train,X_test,y_test,mytitle)
	myutil.plot_learning_curve(MLPClassifier(solver=solver,hidden_layer_sizes=hidden_layer_sizes,activation=activation,alpha=alpha,max_iter=10000),X,y,mytitle)
	myutil.show_pic(mytitle)
	mlp = MLPClassifier(solver=solver,hidden_layer_sizes=hidden_layer_sizes,activation=activation,alpha=alpha,max_iter=10000).fit(X1,y)
	myutil.draw_scatter_for_clf(X1,y,mlp,mytitle)
				   
def MLPClassifier_base():
	mydatas = [datasets.load_iris(), datasets.load_wine(), datasets.load_breast_cancer()]
	titles = ["鸢尾花数据","红酒数据","乳腺癌数据"]
	for (mydata,title) in zip(mydatas, titles):
		ten = [10]
		hundred = [100]
		two_ten = [10,10]
		Parameters = [['lbfgs',hundred,'relu',1,0.0001],
			      ['lbfgs',ten,'relu',1,0.0001],
			      ['lbfgs',two_ten,'relu',2,0.0001],
			      ['lbfgs',two_ten,'tanh',2,0.0001],
			      ['lbfgs',two_ten,'tanh',2,1]]
		for Parameter in Parameters:
			My_MLPClassifier(Parameter[0],Parameter[1],Parameter[2],Parameter[3],Parameter[4],mydata,title)

def writeing():
	minist = datasets.fetch_openml('mnist_784')
	print("样本数量{}，样本特征数：{}".format(minist.data.shape[0],minist.data.shape[1]))
	X = minist.data/255
	y = minist.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=62)
	mlp = MLPClassifier(solver='lbfgs',hidden_layer_sizes=[100,100],activation='tanh',alpha=1e-5,random_state=62)
	mlp.fit(X_train,y_train)
	print("测试集得分:{:.2%}".format(mlp.score(X_test,y_test)))
	image=Image.open('9.png').convert('F')
	# 调整图像的大小
	image=image.resize((28,28))
	arr=[]
	# 将图像中的像素作为预测数据点的特征
	for i in range(28):
		for j in range(28):
			pixel=1.0-float(image.getpixel((j,i)))/255
			arr.append(pixel)
	# 由于只有一个样本,所以需要进行reshape操作
	arr1=np.array(arr).reshape(1,-1) # reshape成一行, 无论多少列
	# 进行图像识别
	plt.imshow(image)
	plt.show()
	print('图片中的数字是:{}'.format(mlp.predict(arr1)[0]))
	
##########################################################################################################################
#MLPRegressor
##########################################################################################################################
def MLPRegressor_make_regression():
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = datasets.make_regression(n_samples=100,n_features=1,n_informative=2,noise=50,random_state=8)
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = MLPRegressor(max_iter=20000).fit(X,y)
	title = "MLPRegressor make_regression数据集(有噪音)"
	myutil.draw_line(X[:,0],y,clf,title)

def My_MLPRegressor(solver,hidden_layer_sizes,activation,level,alpha,mydata,title):
	warnings.filterwarnings("ignore")
	myutil = util()
	X,y = mydata.data,mydata.target
	X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=8,test_size=0.3)
	clf = MLPRegressor(solver=solver,hidden_layer_sizes=hidden_layer_sizes,activation=activation,alpha=alpha,max_iter=10000).fit(X_train,y_train)
	mytitle = "MLPRegressor("+title+"):solver:"+solver+",node:"+str(hidden_layer_sizes)+",activation:"+activation+",level="+str(level)+",alpha="+str(alpha)
	myutil.print_scores(clf,X_train,y_train,X_test,y_test,mytitle)

def MLPRegressor_base():
	mydatas = [datasets.load_diabetes(), datasets.load_boston()]
	titles = ["糖尿病数据","波士顿房价数据"]
	for (mydata,title) in zip(mydatas, titles):
		ten = [10]
		hundred = [100]
		two_ten = [10,10]
		Parameters = [['lbfgs',hundred,'relu',1,0.0001],
			      ['lbfgs',ten,'relu',1,0.0001],
			      ['lbfgs',two_ten,'relu',2,0.0001],
			      ['lbfgs',two_ten,'tanh',2,0.0001],
			      ['lbfgs',two_ten,'tanh',2,1]]
		for Parameter in Parameters:
			My_MLPRegressor(Parameter[0],Parameter[1],Parameter[2],Parameter[3],Parameter[4],mydata,title)
			
if __name__=="__main__":
	#relu_and_tanh()
	#MLPClassifier_base()
	#writeing()
	#MLPRegressor_make_regression()
	MLPRegressor_base()
	

