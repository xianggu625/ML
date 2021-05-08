# coding:utf-8
import numpy as np

def create_array():
    a = np.empty([3,2], dtype = int) 
    print("随机整数数组:\n",a)

    a = np.zeros([2,2], dtype = float) 
    print("全0数组:\n",a)

    a = np.ones([2,2], dtype = int) 
    print("全1数组:\n",a)

    a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]) 
    print("创建数组:\n",a)

    x = [(1,2,3,4),(5,6,7,8)]
    a = np.asarray(x) 
    print("从原来的数组或元组创建数组:\n",a)

    a = np.random.random([2,3])
    print("产生随机数组:\n",a)

    a = np.linspace(start=0,stop=20,num=11) 
    print("产生一维等差数组:\n",a)

    a = np.logspace(start=1,stop=3,num=3,endpoint=True,base=2) 
    print("产生[base^start,base^stop] 个数组:\n",a)

def get_attribute():
    a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    print("a的行列数:\n",a.shape)
    print("a的数据类型:\n",a.dtype)
    print("a的秩:\n",a.ndim)
    print("a的总个数:\n",a.size)

def np_operator():
    a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    print("a:\n",a)
    print("a数组调整:\n",a.reshape(3,4))
    print("a前2行:\n",a[0:2])
    print("a第3列:\n",a[:,2])
    print("a每个元素+1:\n",a+1)
    print("a每个元素平方:\n",a**2)
    print("判断a每个元素是否等于5:\n",a==5)
    print("a列求和:\n",a.sum(axis=0))
    print("a行求和:\n",a.sum(axis=1))
    A = np.array([[1,1],[2,1]])
    B = np.array([[1,2],[3,4]])
    print("A:\n",A)
    print("B:\n",B)
    print("矩阵乘法:\n",A.dot(B))
    print("矩阵乘法:\n",np.dot(A,B))
    print("矩阵点乘:\n",A*B)
    print("A的扩展:\n",np.tile(A,(2,3)))


def ord_index():
    a = np.array([[1,3,2],[6,5,4],[7,8,9],[11,10,12]])
    print("a:\n",a)
    index = a.argmax(axis=0)
    print("a中每一列最大值索引:\n",index)
    print("a中每一列最大值:\n",a[index,range(a.shape[1])])
    print("a中每行元素从小到大排序:\n",np.sort(a,axis=1))
    print("返回数组从小到大的索引值:\n",np.argsort(a))

def load_file():
    a = range(10)
    np.save('testfile',a)
    b = np.load('testfile.npy')
    print(b)
    a = range(15)
    np.savetxt('mydata.txt',a)
    b = np.loadtxt('mydata.txt')
    print(b) 
    

    
if __name__=="__main__":
    #create_array()
    #get_attribute()
    np_operator()
    #ord_index()
    #load_file()
