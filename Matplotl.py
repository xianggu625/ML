# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#1.折线
def broken_line():
    x = np.linspace(0,10,1000)
    y = np.sin(x)
    z = np.cos(x)
    plt.style.use('seaborn-whitegrid')#图像风格
    flg =plt.figure(figsize=(8,4))#创建图大小
    plt.rcParams["font.family"] = 'Arial Unicode MS'#设置字体
    plt.title('sin(x) & cos(x)')#设置标题
    plt.xlabel('x')#设置x轴标题
    plt.ylabel('y')#设置y轴标题
    plt.xlim(-2.5,12.5)#设置x轴上下限
    plt.ylim(-2,3)#设置y轴上下限

    sin_line = plt.plot(x,y,label='$\sin(x)$',color='blue',linestyle='dashdot')
    cos_line = plt.plot(x,z,label='$\cos(x)$',color=(0.2,0.8,0.3),linestyle='--')
    plt.plot(x,y+1,":c",label='$\sin(x)+1$')

    plt.legend(loc='upper left',frameon=True)
    plt.show()

#多子图
def Multiple():
    figure, ax = plt.subplots()
    figure.suptitle('Subplots demo')

    #误差线
    x = np.linspace(0,10,50)
    data1 = np.sin(x)+x*0.48
    plt.subplot(2,2,1)
    plt.errorbar(x,data1,yerr=x*0.48,fmt='.k',ecolor='green')

    #饼图
    data2 = [0,1,0,4,0.3,0.2]
    plt.subplot(2,2,2)
    plt.pie(data2)

    #等高线
    x = np.linspace(0,5,50)
    y = np.linspace(0,5,40)
    x,y = np.meshgrid(x,y)
    plt.subplot(2,2,3)
    plt.contour(x,y,f(x,y),colors='blue')

    #直方图
    data4 = np.random.rand(1000)
    plt.subplot(2,2,4)
    plt.hist(data4)
    plt.show()
    

def f(x,y):
    return np.sin(x) ** 10 + np.cos(x*y+12)*np.cos(x)

#对数几率方程
def logistic_function():
    z = np.linspace(-10,10,100)
    y = 1/(1+math.e**(-1*z))
    plt.style.use('seaborn-whitegrid')#图像风格
    flg =plt.figure(figsize=(8,4))#创建图大小
    plt.title('Logistic Function')#设置标题
    plt.xlabel('z')#设置x轴标题
    plt.ylabel('y')#设置y轴标题
    plt.xlim(-10,10)#设置x轴上下限
    plt.ylim(0,1)#设置y轴上下
    plt.plot(z,y,color='blue',linestyle='dashdot')
    plt.show()

    
if __name__=="__main__":
    #broken_line()
    #Multiple()
    logistic_function()
