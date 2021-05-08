# coding:utf-8
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from PIL import Image
from sklearn.model_selection import learning_curve,KFold

class util:
	#定义一个函数来画图
	def make_meshgrid(self,x, y, h=.02):
		x_min,x_max = x.min()-1,x.max()+1
		y_min,y_max = y.min()-1,y.max()+1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
		return xx,yy

	#定义一个绘制等高线的函数
	def plot_contour(self,ax, clf, xx,yy,**params) :
		Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
		Z = Z.reshape(xx.shape)
		out = ax.contourf(xx,yy,Z,**params)
		return out
	
	#显示图片
	def show_pic(self,title):
		plt.rcParams['font.sans-serif']=['SimHei']
		plt.rcParams['axes.unicode_minus']=False
		plt.suptitle(title)
		plt.show()
	
	#画拟合线
	def draw_line(self,X,y,clf,title):
		Z = np.linspace(-3,3,100)
		plt.scatter(X,y,c='b',s=60)
		try:
			plt.plot(Z,clf.predict(Z.reshape(-1,1)),c='k')
		except Exception as e:
			print("不支持画线")
		self.show_pic(title)


	#画一个算法的散点图
	def draw_scatter(self,X, y,clf,title):
		plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolor='k')
		x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
		y_min,y_max = X[:,1].min()-1,X[:,1].max()+1
		xx,yy = np.meshgrid(np.arange(x_min,x_max,.02),
				    np.arange(y_min,y_max,.02))#生成网格点坐标矩阵
		Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])#预测数据集X的结果
		Z = Z.reshape(xx.shape) 
		plt.pcolormesh(xx, yy, Z,shading='auto',cmap=plt.cm.Spectral)#绘制分类图
		plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolor='k')
		plt.xlim(xx.min(),xx.max())#设置或查询 x 轴限制
		plt.ylim(yy.min(),yy.max())#设置或查询 y 轴限制
		self.show_pic(title)

	#画普通算法的散点图
	def draw_scatter_for_clf(self,X,y,clf,title):
		X0, X1 = X[:,0],X[:,1]
		xx, yy = self.make_meshgrid(X0, X1) 
		self.plot_contour(plt,clf,xx,yy,cmap=plt.cm.plasma,alpha=0.8)
		plt.scatter(X0,X1,c=y,cmap=plt.cm.plasma,s=20,edgecolors='k')
		plt.xlabel(u'特征0')
		plt.ylabel(u'特征1')
		plt.xticks(())
		plt.yticks(())
		plt.suptitle(title)
		self.show_pic(title)
		

	#画聚类散点图
	def draw_scatter_for_Clustering(self,X,y,result,title,algorithm):
		if len(y)!=0:
			print(title+"原始数据集分配簇标签为：\n{}".format(y))
			print(title+" "+algorithm+" 训练簇标签为：\n{}".format(result))
			plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolor='k')
			self.show_pic(title+"原始数据集分配簇标签图")
		plt.scatter(X[:,0],X[:,1],c=result,cmap=plt.cm.spring,edgecolor='k')
		self.show_pic(title+" "+algorithm+"训练簇标签标签图")

	#画降维散点图
	def draw_scatter_for_Dimension_Reduction(self,X,mydata,title,algorithm):
		# 将3个分类主成分提取出来
		X0 = X[mydata.target==0]
		X1 = X[mydata.target==1]
		X2 = X[mydata.target==2]
		#绘制散点图
		plt.scatter(X0[:,0],X0[:,1],c='r',s=60,edgecolor='k')
		plt.scatter(X1[:,0],X1[:,1],c='g',s=60,edgecolor='k')
		plt.scatter(X2[:,0],X2[:,1],c='b',s=60,edgecolor='k')
		#设置图注
		plt.legend(mydata.target_names,loc='best')
		plt.xlabel('特征 1')
		plt.ylabel('特征 2')
		mytitle = title+algorithm+"散点图"
		self.show_pic(mytitle)

        #画热力图
	def draw_Heat_chart(self,pca,mydata,title,algorithm):
                plt.matshow(pca.components_,cmap='plasma')
                #纵轴为主成分
                plt.yticks([0,1],['特征 1','特征 2'])
                plt.colorbar()
                #横轴为原始特征向量
                plt.xticks(range(len(mydata.feature_names)),mydata.feature_names,rotation=60,ha='left')
                mytitletitle = title+algorithm+"热度图"
                self.show_pic(mytitletitle)
                
        
	#定义一个绘制学习曲线的函数
	def plot_learning_curve(self,est, X, y,title):#learning_curve:学习曲线
		tarining_set_size,train_scores,test_scores = learning_curve(
			est,X,y,train_sizes=np.linspace(.1,1,20),cv=KFold(20,shuffle=True,random_state=1))
		estimator_name = est.__class__.__name__
		line = plt.plot(tarining_set_size,train_scores.mean(axis=1),'o-',label=u'训练得分'+estimator_name,c='r')
		plt.plot(tarining_set_size,test_scores.mean(axis=1),'o-',label=u'测试得分'+estimator_name,c='g')
		plt.grid()
		plt.xlabel(u'训练设置值')
		plt.ylabel(u"得分")
		plt.ylim(0,1.1)
		plt.legend(loc='lower right')

	#打印得分
	def print_scores(self,clf,X_train,y_train,X_test,y_test,title):
		title = title+":\n{:.2%}"
		print(title.format(clf.score(X_train,y_train)))
		print(title.format(clf.score(X_test,y_test)))
		

