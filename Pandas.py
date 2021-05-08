# coding:utf-8
import numpy as np
import pandas as pd

def init():
    df = pd.DataFrame(np.array([[1,2],[3,4],[5,6],[7,8]]),columns=['A','B'],index=[3,4,5,6])
    return df

def create_data():
    # 使用列表创建序列
    print("1 用列表创建序列")
    s1 = pd.Series([1,2,3,4,5])
    print("s1内容:\n",s1)
    print("s1索引:\n",s1.index)
    print("s1值:\n",s1.values)
    print("2 由字典创建序列")
    dic = {"A":1,"B":2,"C":3,"D":4,"E":5}
    s2 = pd.Series(dic,index=["A","C","E"])
    print("s2:\n",s2)
    print("3 由字典创建序列")
    df1 = pd.DataFrame(s1,columns=["number"]) #指定列名
    print("DataFrame1:\n",df1)
    print("4 通过序列对象产生DataFrame")
    df2 = pd.DataFrame({'A':1,'B':s1,'C':pd.Timestamp('20201208'),'D':'hello'})
    print("DataFrame2:\n",df2)
    print("5 通过numpy产生DataFrame")
    df3 = pd.DataFrame(np.array([[1,2],[3,4],[5,6],[7,8]]),columns=['A','B'],index=[3,4,5,6])
    print("DataFrame3:\n",df3)
    print("6 由字典创建序列")
    dic = {"A":1,"B":2,"C":3,"D":4,"E":5}
    s2 = pd.Series(dic,index=["A","C","E"])
    print("s2:\n",s2)
    print("7 通过序列创建DataFrame")
    df1 = pd.DataFrame(s1,columns=["number"]) #指定列名
    print("DataFrame1:\n",df1)

    
def view_data(df):
    print("DataFrame:概要信息\n",df.describe())
    print("DataFrame:头部\n",df.head())
    print("DataFrame 尾部:\n",df.tail(2))
    print("DataFrame 索引:\n",df.index)
    print("DataFrame 列名:\n",df.columns)
    return df

def sort_df(df):
    print("按索引排序:\n",df.sort_index(axis=1,ascending=False))
    print("按值排序:\n",df.sort_values(by='B',ascending=False))


def get_value(df):
    print("原数据:\n",df)
    print("按列获取内容:\n",df['A'])
    print("切片操作:\n",df[0:3])
    print("基于行列标签获取数据（loc）:\n",df.loc[:4,:'C'])
    print("基于行列索引获取数据（iloc）:\n",df.iloc[:2,:2])

def data_oper(df):
    print("原数据:\n",df)
    print("每个字均+1:\n",df.add(1))
    print("数据每一列均值:\n",df.mean())
    print("数据每一行均值:\n",df.mean(1))
    print("apply函数:\n",df.apply(lambda x:x.max()-x.min()))


def load_file():
    data = pd.read_csv('Pandas.csv')
    print("Pandas.csv:\n",data)
    data.to_csv('Pandas.csv',index=False)
    #index=False 不把索引写进文件中
    data = pd.read_excel('Pandas.xlsx','Sheet1')
    print("my.xlsx:\n",data)
    #pip3 install openpyxl
    data.to_excel('Pandas.xlsx',sheet_name='Sheet1',index=False) 
        
if __name__=="__main__":
    #df = init()
    #create_data()
    #view_data(df)
    #sort_df(df)
    #get_value(df)
    #data_oper(df)
    load_file()
