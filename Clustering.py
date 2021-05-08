# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs 
from sklearn import datasets
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.preprocessing import StandardScaler
import mglearn #pip3 install mglearn
from scipy.cluster.hierarchy import dendrogram,ward 
from Util import util

###############################################################################################################################################
#K均值(KMeans)算法
###############################################################################################################################################
def KMeans_for_blobs():
        blobs = make_blobs(random_state=1,centers=1)
        X = blobs[0]
        y = blobs[1]
        #设置簇个数为3
        Kmeans = KMeans(n_clusters=3)
        Kmeans.fit(X)
        print("训练集数据集分配簇标签为：\n{}".format(Kmeans.labels_))
        print("对训练集数据集预测结果为:",Kmeans.predict(X))
        #画出聚类后的数据集图像
        mglearn.discrete_scatter(X[:,0], X[:,1],Kmeans.labels_,markers='o')
        mglearn.discrete_scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],[0,1,2],markers='^',markeredgewidth=2)
        plt.show()
        X_blobs = blobs[0]
        X_min,X_max = X_blobs[:,0].min()-0.5,X_blobs[:,0].max()+0.5
        y_min,y_max = X_blobs[:,1].min()-0.5,X_blobs[:,1].max()+0.5
        xx, yy = np.meshgrid(np.arange(X_min, X_max, .02),np.arange(y_min, y_max, .02))
        Z = Kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.imshow(Z,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap=plt.cm.summer,aspect='auto',origin='lower')
        plt.plot(X_blobs[:,0],X_blobs[:,1],'r,',markersize=5)
        #用蓝色×代表聚类的中心
        centroids = Kmeans.cluster_centers_
        plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=3,color='b',zorder=10)
        plt.xlim(X_min,X_max)
        plt.ylim(y_min,y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        mglearn.plots.plot_kmeans_boundaries()
        plt.show()

def mglearn_for_plot_kmeans_algorithm():
    mglearn.plots.plot_kmeans_algorithm()
    plt.show()
        
def KMeans_for_iris():
        myutil = util()
        X,y = datasets.load_iris().data,datasets.load_iris().target
        Kmeans = KMeans(n_clusters=3)
        Kmeans.fit(X)
        result = Kmeans.fit_predict(X)
        title = "鸢尾花"
        myutil.draw_scatter_for_Clustering(X,y,result,title,"KMeans")

def KMeans_for_wine():
        myutil = util()
        X,y = datasets.load_wine().data,datasets.load_wine().target
        Kmeans = KMeans(n_clusters=3)
        Kmeans.fit(X)
        result = Kmeans.fit_predict(X)
        title = "红酒"
        myutil.draw_scatter_for_Clustering(X,y,result,title,"KMeans")

def KMeans_for_breast_cancer():
        myutil = util()
        X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
        Kmeans = KMeans(n_clusters=2)
        Kmeans.fit(X)
        result = Kmeans.fit_predict(X)
        title = "乳腺癌"
        myutil.draw_scatter_for_Clustering(X,y,result,title,"KMeans")

#两个月亮
def KMeans_for_two_moon():
        myutil = util()
        X, y = datasets.make_moons(n_samples=200,noise=0.05, random_state=0)
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        # 打印处理后的数据形态
        print("处理后的数据形态:",X_scaled.shape)
        # 处理后的数据形态: (200, 2) 200个样本 2类    
        Kmeans = KMeans(n_clusters=2)
        result=Kmeans.fit_predict(X_scaled)
        title = "两个月亮"
        #绘制簇分配结果
        myutil.draw_scatter_for_Clustering(X,y,result,title,"KMeans")
###############################################################################################################################################
#凝聚算法
###############################################################################################################################################   
def agglomerative_algorithm():
    mglearn.plots.plot_agglomerative_algorithm()
    plt.show()
    blobs = make_blobs(random_state=1,centers=1)
    x_blobs = blobs[0]
    #使用连线方式进行可视化
    linkage =ward(x_blobs)
    dendrogram(linkage)
    ax = plt.gca() # gca：Get Current Axes
    #设定横纵轴标签
    plt.xlabel("sample index")
    plt.ylabel("Cluster distance")
    plt.show()

def AgglomerativeClustering_for_blobs():
        blobs = make_blobs(random_state=1,centers=1)
        X = blobs[0]
        y = blobs[1]
        #设置簇个数为3
        AC = AgglomerativeClustering(n_clusters=3)
        result = AC.fit_predict(X)
        print("训练集数据集分配簇标签为：\n{}".format(AC.labels_))
        print("对训练集数据集预测结果为：\n{}".format(result))
        #画出聚类后的数据集图像
        mglearn.discrete_scatter(X[:,0], X[:,1],AC.labels_,markers='o')
        plt.show()

def AgglomerativeClustering_for_iris():
        myutil = util()
        X,y = datasets.load_iris().data,datasets.load_iris().target
        AC = AgglomerativeClustering(n_clusters=3)
        AC.fit(X)
        result = AC.fit_predict(X)
        title = "鸢尾花"
        myutil.draw_scatter_for_Clustering(X,y,result,title,"凝聚算法")

def AgglomerativeClustering_for_wine():
        myutil = util()
        X,y = datasets.load_wine().data,datasets.load_wine().target
        AC = AgglomerativeClustering(n_clusters=3)
        AC.fit(X)
        result = AC.fit_predict(X)
        title = "红酒"
        myutil.draw_scatter_for_Clustering(X,y,result,title,"凝聚算法")

def AgglomerativeClustering_for_breast_cancer():
        myutil = util()
        X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
        AC = AgglomerativeClustering(n_clusters=2)
        AC.fit(X)
        result = AC.fit_predict(X)
        title = "乳腺癌"
        myutil.draw_scatter_for_Clustering(X,y,result,title,"凝聚算法")

#两个月亮
def AgglomerativeClustering_for_two_moon():
        myutil = util()
        X, y = datasets.make_moons(n_samples=200,noise=0.05, random_state=0)
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        # 打印处理后的数据形态
        print("处理后的数据形态:",X_scaled.shape)
        # 处理后的数据形态: (200, 2) 200个样本 2类    
        AC = AgglomerativeClustering(n_clusters=2)
        result=AC.fit_predict(X_scaled)
        title = "两个月亮"
        #绘制簇分配结果
        myutil.draw_scatter_for_Clustering(X,y,result,title,"凝聚算法")
###############################################################################################################################################
#DBSCAN算法
###############################################################################################################################################
def dbscan_for_blobs():
        myutil = util()
        epss=[0.5,2,0.5]
        min_sampless=[5,5,20]
        for (eps,min_samples) in zip(epss,min_sampless):
                db = DBSCAN(eps=eps,min_samples=min_samples)
                blobs = make_blobs(random_state=1,centers=1)
                X = blobs[0]
                clusters = db.fit_predict(X)
                title = "eps="+str(eps)+",min_samples="+str(min_samples)
                myutil.draw_scatter_for_Clustering(X,"",clusters,title,"DBSCAN")

def dbscan_for_iris():
        myutil = util()
        X,y = datasets.load_iris().data,datasets.load_iris().target
        dbscan = DBSCAN(min_samples=0.5,eps=1)
        dbscan.fit(X)
        result = dbscan.fit_predict(X)
        title = "鸢尾花"
        myutil.draw_scatter_for_Clustering(X,y,result,title,"DBSCAN")

def dbscan_for_wine():
        myutil = util()
        X,y = datasets.load_wine().data,datasets.load_wine().target
        dbscan = DBSCAN(min_samples=0.5,eps=50)
        dbscan.fit(X)
        result = dbscan.fit_predict(X)
        title = "红酒"
        myutil.draw_scatter_for_Clustering(X,y,result,title,"DBSCAN")

def dbscan_for_breast_cancer():
        myutil = util()
        X,y = datasets.load_breast_cancer().data,datasets.load_breast_cancer().target
        dbscan = DBSCAN(min_samples=0.5,eps=100)
        dbscan.fit(X)
        result = dbscan.fit_predict(X)
        title = "乳腺癌"
        myutil.draw_scatter_for_Clustering(X,y,result,title,"DBSCAN")

#两个月亮
def dbscan_for_two_moon():
        myutil = util()
        X, y = datasets.make_moons(n_samples=200,noise=0.05, random_state=0)
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        # 打印处理后的数据形态
        print("处理后的数据形态:",X_scaled.shape)
        # 处理后的数据形态: (200, 2) 200个样本 2类    
        dbscan = DBSCAN()
        result=dbscan.fit_predict(X_scaled)
        title = "两个月亮"
        #绘制簇分配结果
        myutil.draw_scatter_for_Clustering(X,y,result,title,"DBSCAN")

if __name__=="__main__":
        # KMeans_for_blobs()
        # mglearn_for_plot_kmeans_algorithm()
        # KMeans_for_iris()
        # KMeans_for_wine()
        # KMeans_for_breast_cancer()
        # KMeans_for_two_moon()
        # AgglomerativeClustering_for_blobs()
        # AgglomerativeClustering_for_iris()
        # AgglomerativeClustering_for_wine()
        # AgglomerativeClustering_for_breast_cancer()
        # AgglomerativeClustering_for_two_moon()
        # dbscan_for_blobs()
        # dbscan_for_iris()
        # dbscan_for_wine()
        # dbscan_for_breast_cancer()
        # dbscan_for_two_moon()
