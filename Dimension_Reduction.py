# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA,NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from Util import util

##################################################################
#PCA
#################################################################
#主成分提取PCA可视化
def dimension_reduction_for_pca(mydata,title):
        myutil = util()
        scaler = StandardScaler()
        X,y = mydata.data,mydata.target
        #由于是无监督学习，所以尽对X进行拟合
        X_scaled = scaler.fit_transform(X)
        # 打印处理后的数据形态
        print("处理后的数据形态:",X_scaled.shape)
        # 进行PCA处理
        pca = PCA(n_components=2) #降到2类
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        # 打印主成分提取后的数据形态
        print("主成分提取后的数据形态:",X_pca.shape)
        myutil.draw_scatter_for_Dimension_Reduction(X_pca,mydata,title,'主成分提取(PCA)')
        #使用主成分绘制热度图
        myutil.draw_Heat_chart(pca,mydata,title,'主成分提取(PCA)')

def call_dimension_reduction_for_pca():
        mydatas = [datasets.load_iris(),datasets.load_wine(),datasets.load_breast_cancer()]
        titles = ["鸢尾花","红酒","乳腺癌"]
        for (mydata,title) in zip(mydatas,titles):
                dimension_reduction_for_pca(mydata,title)

def pca_for_face():
        faces = datasets.fetch_lfw_people(min_faces_per_person=20,resize=0.8)
        image_shape = faces.images[0].shape
        #把照片打印出来
        fig, axes = plt.subplots(3,4,figsize=(12,9),subplot_kw={'xticks':(),'yticks':()})
        for target,image,ax in zip(faces.target,faces.images,axes.ravel()):
                ax.imshow(image,cmap=plt.cm.gray)
                ax.set_title(faces.target_names[target])
        plt.show()
        #用神经网络模型进行训练
        X_train,X_test,y_train,y_test = train_test_split(faces.data/255,faces.target,random_state=62)
        mlp = MLPClassifier(hidden_layer_sizes=[100,100],random_state=62,max_iter=400)
        mlp.fit(X_train,y_train)
        print("模型识别准确率:{:.2%}".format(mlp.score(X_test,y_test)))
        #使用白化功能处理人脸数据
        pca = PCA(whiten=True,n_components=0.9,random_state=62).fit(X_train)
        X_train_whiten = pca.transform(X_train)
        X_test_whiten = pca.transform(X_test)
        print("白化后数据形态:{}".format(X_train_whiten.shape))
        #使用白化后的神经网络训练
        mlp.fit(X_train_whiten,y_train)
        print("白化后模型识别准确率:{:.2%}".format(mlp.score(X_test_whiten,y_test)))

##################################################################
#NMF
#################################################################
#非负矩阵分解NMF可视化
def dimension_reduction_for_nmf(mydata,title):
        myutil = util()
        X,y = mydata.data,mydata.target
        #由于是无监督学习，所以尽对X进行拟合
        # 打印处理后的数据形态
        print("处理后的数据形态:",X.shape)
        # 进行PCA处理
        nmf = NMF(n_components=2,random_state=62,init='nndsvdar',max_iter=10000) #降到2类
        nmf.fit(X)
        X_nmf = nmf.transform(X)
        # 打印主成分提取后的数据形态
        print("非负矩阵分解后的数据形态:",X_nmf.shape)
        myutil.draw_scatter_for_Dimension_Reduction(X_nmf,mydata,title,'非负矩阵分解(NMF)')
        #使用主成分绘制热度图
        myutil.draw_Heat_chart(nmf,mydata,title,'非负矩阵分解(NMF)')

def call_dimension_reduction_for_nmf():
        mydatas = [datasets.load_iris(),datasets.load_wine(),datasets.load_breast_cancer()]
        titles = ["鸢尾花","红酒","乳腺癌"]
        for (mydata,title) in zip(mydatas,titles):
                dimension_reduction_for_nmf(mydata,title)

def nmf_for_face():
        faces = datasets.fetch_lfw_people(min_faces_per_person=20,resize=0.8)
        #用NMF模型进行模拟
        X_train,X_test,y_train,y_test = train_test_split(faces.data/255,faces.target,random_state=62)
        mlp = MLPClassifier(hidden_layer_sizes=[100,100],random_state=62,max_iter=10000)
        nmf = NMF(n_components=105,random_state=62,init='nndsvdar').fit(X_train)#NMF中n_components不支持浮点数
        X_train_nmf = nmf.transform(X_train)
        X_test_nmf = nmf.transform(X_test)
        print("NMF处理后数据形态:{}".format(X_train_nmf.shape))
        #用神经网络模型进行训练
        mlp.fit(X_train_nmf,y_train)
        print("NMF训练后模型识别准确率:{:.2%}".format(mlp.score(X_test_nmf,y_test)))

##################################################################
#LinearDiscriminantAnalysis
#################################################################
#线性判别分析
def dimension_reduction_for_lda(mydata,title):
        myutil = util()
        X,y = mydata.data,mydata.target
        print("处理后的数据形态:",X.shape)
        # 进行PCA处理
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(X,y)
        X_lda = lda.transform(X)
        # 打印主成分提取后的数据形态
        print("非负矩阵分解后的数据形态:",X_lda.shape)
        myutil.draw_scatter_for_Dimension_Reduction(X_lda,mydata,title,'线性判别分析(LDA)')

def call_dimension_reduction_for_lda():
        mydatas = [datasets.load_iris(),datasets.load_wine(),datasets.load_breast_cancer()]
        titles = ["鸢尾花","红酒"]
        for (mydata,title) in zip(mydatas,titles):
                dimension_reduction_for_lda(mydata,title)
   
if __name__=="__main__":
        #call_dimension_reduction_for_pca()
        #pca_for_face()
        call_dimension_reduction_for_nmf()
        #nmf_for_face()
        #call_dimension_reduction_for_lda()