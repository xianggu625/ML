# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs,make_classification,make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
	
def MyBlobs():
        X,y = make_blobs(n_samples=20,centers=2, random_state=8)
        print("X is :\n",X)
        print("y is :\n",y)
        plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolor='k')
        plt.show()
        
def MyClassification():
        X,y= make_classification(n_samples=20 ,random_state=8)
        print("X is :\n",X)
        print("y is :\n",y)
        
def MyRegression():
        X,y= make_regression(n_features=2,random_state=8)
        print("X is :\n",X)
        print("y is :\n",y)

	
if __name__=="__main__":
        #MyBlobs()
        #MyClassification()
        MyRegression()

	

	
		
