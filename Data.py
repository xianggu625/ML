# coding:utf-8
import numpy as np
import pandas as pd
#C:\Users\xiang>pip3 install pandas

#1 数据清洗和准备
#1.1 数据概览和类型转换
def data_info():
    data = pd.read_csv('my.csv')
    print("data.info:\n",data.info())
    print("data.shape:\n",data.shape)#规模
    print("data.dtype:\n",data.dtypes)#类型
    print("data.head:\n",data.head())#前五行
    print("data.tail:\n",data.tail())#后五行
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   编号      3 non-null      int64 
 1   姓名      3 non-null      object
 2   手机      3 non-null      int64 
 3   Email   3 non-null      object
dtypes: int64(2), object(2)
memory usage: 224.0+ bytes
data.info:
 None
data.shape:
 (3, 4)
data.dtype:
 编号        int64
姓名       object
手机        int64
Email    object
dtype: object
data.head:
    编号  姓名           手机             Email
0   1  张三  13686544556  zhangsan@126.com
1   2  李四  13554346655      lisi@126.com
2   3  王五  18954357788    wangwu@126.com
data.tail:
    编号  姓名           手机             Email
0   1  张三  13686544556  zhangsan@126.com
1   2  李四  13554346655      lisi@126.com
2   3  王五  18954357788    wangwu@126.com
'''

def change_type():
    data = pd.read_csv('user.csv')
    data['编号'] = data['编号'].astype(str)
    data['价格'] = data['价格'].str[1:].astype(float)
    data['生日'] = pd.to_datetime(data['生日'],format='%Y-%m-%d')
    print("data.dtype:\n",data.dtypes)
'''
data.dtype:
编号               object
姓名               object
手机                int64
Email            object
生日       datetime64[ns]
价格              float64
dtype: object
'''

#1.2 处理丢失数据
def judge_is_null():
    data = pd.read_csv('my2.csv')
    print("data is:\n",data)
    print("data is null\n",data.isnull())
    print("去除缺省值的数据行",data.dropna())
    print("去除缺省值的数据列",data.dropna(axis=1))
'''
data is:
    编号  姓名           手机             Email
0   1  张三  13686544556  zhangsan@126.com
1   2  李四  13554346655               NaN
2   3  王五  18954357788    wangwu@126.com
data is null
       编号     姓名     手机  Email
0  False  False  False  False
1  False  False  False   True
2  False  False  False  False
去除缺省值的数据行    编号  姓名           手机             Email
0   1  张三  13686544556  zhangsan@126.com
2   3  王五  18954357788    wangwu@126.com
去除缺省值的数据列    编号  姓名           手机
0   1  张三  13686544556
1   2  李四  13554346655
2   3  王五  18954357788 
'''

def replace_null():
    data = pd.read_csv('my2.csv')
    print("用0填充",data.fillna(0))
    print("用字典填充",data.fillna({"姓名":"--","手机":"未知","Email":"---"}))
'''
用0填充    编号  姓名           手机             Email
0   1  张三  13686544556  zhangsan@126.com
1   2  李四  13554346655                 0
2   3  王五  18954357788    wangwu@126.com
用字典填充    编号  姓名           手机             Email
0   1  张三  13686544556  zhangsan@126.com
1   2  李四  13554346655               ---
2   3  王五  18954357788    wangwu@126.com
'''

#1.3 处理重复数据
def is_duplicate():
    data = pd.DataFrame({"A":['a','b']*3+['a','b'],"B":[1,1,2,2,3,3,2,3]})
    print("data is:\n",data)
    print("数据是否有重复:\n",data.duplicated())
    print("A列数据是否有重复:\n",data.duplicated(['A']))
'''
data is:
   A  B
0  a  1
1  b  1
2  a  2
3  b  2
4  a  3
5  b  3
6  a  2
7  b  3
数据是否有重复:
0    False
1    False
2    False
3    False
4    False
5    False
6     True
7     True
dtype: bool
A列数据是否有重复:
0    False
1    False
2     True
3     True
4     True
5     True
6     True
7     True
dtype: bool
'''
def drop_duplicate():
    data = pd.DataFrame({"A":['a','b']*3+['a','b'],"B":[1,1,2,2,3,3,2,3]})
    print("删除重复行:\n",data.drop_duplicates())
    print("删除A列的重复数据:\n",data.drop_duplicates(['A']))
'''
删除重复行:
   A  B
0  a  1
1  b  1
2  a  2
3  b  2
4  a  3
5  b  3
删除A列的重复数据:
    A  B
0  a  1
1  b  1
'''

#1.4 数据转换
def data_type_transfer():
    data = pd.read_csv('user1.csv')
    data['姓名'] = data['姓名'].str.lower() #姓名转为小写
    data['生日'] = pd.to_datetime(data['生日'],format='%Y-%m-%d')
    city_to_Province ={
    '上海':'上海',
    '北京':'北京',
    '南京':'江苏',
    '银川':'宁夏',
    '苏州':'江苏',
    '无锡':'江苏'
    }
    data['省'] = data['城市'].map(city_to_Province)
    print("Data is:\n",data)
'''
Data is:
    编号      姓名           手机         生日  城市   省
0   1   jerry  13686544556 1972-08-23  上海  上海
1   2     tom  13554346655 1965-06-24  南京  江苏
2   3   cindy  18954357788 1987-04-23  银川  宁夏
3   4   peter  18067543233 1992-04-23  北京  北京
4   5  jessca  13987654567 1957-04-23  南京  江苏
5   6    john  13899773438 1962-04-23  苏州  江苏
6   7   black  18900987654 1973-04-23  无锡  江苏
7   8   white  18811221111 1983-04-23  苏州  江苏
'''

#1.5 数据替换
def data_replace():
    data = pd.Series([1,2,3,4,7,888,9999])
    print("Data is:\n",data)
    print("Replace 9999 to -1\n",data.replace(9999,-1))
    print("Replace 9999 and 888\n",data.replace([9999,888],[-1,0]))
'''
Data is:
0       1
1       2
2       3
3       4
4       7
5     888
6    9999
dtype: int64
Replace 9999 to -1
    0      1
1      2
2      3
3      4
4      7
5    888
6     -1
dtype: int64
Replace 9999 and 888
0    1
1    2
2    3
3    4
4    7
5    0
6   -1
dtype: int64
'''

#1.6 数据离散化，数据拆分
def data_categories():
    scores = [54,63,76,83,93]
    grade = [0,60,70,80,90,100]
    data = pd.cut(scores,grade)
    print("data is:\n",data)
    print("data.categories is:\n",data.categories)
    print("data.codes is:\n",data.codes)
    print("value counts:\n",pd.value_counts(data))
    print("User Lables:\n",pd.cut(scores,grade,labels=['E','D','C','B','A']))
'''
data is:
[(0, 60], (60, 70], (70, 80], (80, 90], (90, 100]]
Categories (5, interval[int64]): [(0, 60] < (60, 70] < (70, 80] < (80, 90] < (90, 100]]
data.categories is:
 IntervalIndex([(0, 60], (60, 70], (70, 80], (80, 90], (90, 100]],
              closed='right',
              dtype='interval[int64]')
data.codes is:
[0 1 2 3 4]
value counts:
(90, 100]    1
(80, 90]     1
(70, 80]     1
(60, 70]     1
(0, 60]      1
dtype: int64
User Lables:
['E', 'D', 'C', 'B', 'A']
Categories (5, object): ['E' < 'D' < 'C' < 'B' < 'A']
'''

#1.7 过滤异常值
def data_filter():
    data = pd.DataFrame(np.random.rand(1000,3))
    data = round(data*100)
    print("data info:\n",data.describe())
    print("data[1]>60:\n",data[1][data[1]>60])#第一列大于60
    print("any data>60:\n",data[(data>60).any(1)])#任意一列大于60
'''
data info:
                  0            1            2
count  1000.000000  1000.000000  1000.000000
mean     49.901000    49.376000    50.897000
std      29.140906    29.287102    29.123918
min       0.000000     0.000000     0.000000
25%      24.000000    25.000000    25.000000
50%      49.500000    47.000000    52.000000
75%      75.000000    76.000000    77.000000
max     100.000000   100.000000   100.000000
data[1]>60:
1      87.0
7      89.0
9      76.0
10     94.0
11     74.0
       ... 
991    90.0
993    80.0
996    76.0
997    61.0
999    88.0
Name: 1, Length: 386, dtype: float64
any data>60:
        0     1     2
0    74.0  32.0  61.0
1     8.0  87.0  38.0
2    65.0   9.0   3.0
3    21.0  35.0  61.0
4    46.0  10.0  97.0
..    ...   ...   ...
994  85.0   7.0  64.0
996  86.0  76.0  60.0
997  51.0  61.0  96.0
998  85.0  38.0  30.0
999  27.0  88.0  75.0

[788 rows x 3 columns]
'''

#1.8 字符串处理
#1.8.1普通处理法
def str_deal():
    text = '1,2,3,4,5,  6'
    splittext = text.split(',')
    striptext = [x.strip() for x in splittext]
    print("split:\n",splittext)
    print("strip:\n",striptext)
    print("join:\n","+".join(striptext))
    print("find:\n",text.find('3'))
    print("replace:\n",text.replace('6','0'))
'''
split:
 ['1', '2', '3', '4', '5', '  6']
strip:
 ['1', '2', '3', '4', '5', '6']
join:
 1+2+3+4+5+6
find:
 4
replace:
 1,2,3,4,5,  0
'''
#1.8.2正则表达式处理（略）
#2 数据规整
#2.1层次化索引
def data_index():
    data = pd.Series(np.random.rand(6),
    index=[
                ['江苏','江苏','浙江','浙江','广东','广东'],
                ['南京','苏州','杭州','宁波','广州','深圳']
            ])
    data = round(data*100)
    print("data is:\n",data)
    print("data index is:\n",data.index)
    print("data['江苏']:\n",data['江苏'])
    print("data.loc[:,'深圳']:\n",data.loc[:,'深圳'])
    print("data.unstack()\n",data.unstack())
    print("data.unstack().stack()\n",data.unstack().stack())
'''
data is:
江苏  南京   -1.621536
      苏州   -0.715182
浙江  杭州    0.143545
      宁波   -0.616146
广东  广州   -0.325306
      深圳   -0.733617
dtype: float64
data index is:
 MultiIndex([('江苏', '南京'),
            ('江苏', '苏州'),
            ('浙江', '杭州'),
            ('浙江', '宁波'),
            ('广东', '广州'),
            ('广东', '深圳')],
           )
data['江苏']:
南京   -1.621536
苏州   -0.715182
dtype: float64
data.loc[:,'深圳']:
广东   -0.733617
dtype: float64
data.unstack()
           南京        宁波        广州        杭州        深圳        苏州
广东       NaN         NaN         -0.325306   NaN         0.733617    NaN
江苏       -1.621536   NaN         NaN         NaN         NaN         -0.715182
浙江       NaN         -0.616146   NaN         0.143545    NaN         NaN
data.unstack().stack()
广东  广州   -0.325306
      深圳   -0.733617
江苏  南京   -1.621536
      苏州   -0.715182
浙江  宁波   -0.616146
      杭州    0.143545
dtype: float64
'''
#2.2 合并数据集
def merge_data(): #按指定列
    df1 = pd.DataFrame({'id':[1,2,3,4],'val':['Jerry','Tom','Kerry','Jessca']})
    df2 = pd.DataFrame({'id':[1,2,3,4,5],'val':['Kerry','Peter','Jerry','Tom','Jessca']})
    print("df1 is:\n",df1)
    print("df2 is:\n",df2)
    print("merage:\n",pd.merge(df1,df2,on='id'))
    df1 = pd.DataFrame({'id1':[1,2,3,4],'val':['Jerry','Tom','Kerry','Jessca']})
    df2 = pd.DataFrame({'id2':[1,2,3,4,5],'val':['Kerry','Peter','Jerry','Tom','Jessca']})
    print("merage:\n",pd.merge(df1,df2,left_on='id1',right_on='id2'))
    print("merage:\n",pd.merge(df1,df2,left_on='id1',right_on='id2',how='outer'))
'''
df1 is:
    id     val
0   1   Jerry
1   2     Tom
2   3   Kerry
3   4  Jessca
df2 is:
    id     val
0   1   Kerry
1   2   Peter
2   3   Jerry
3   4     Tom
4   5  Jessca
merage:
    id   val_x  val_y
0   1   Jerry  Kerry
1   2     Tom  Peter
2   3   Kerry  Jerry
3   4  Jessca    Tom
merage:
    id1   val_x  id2  val_y
0    1   Jerry    1  Kerry
1    2     Tom    2  Peter
2    3   Kerry    3  Jerry
3    4  Jessca    4    Tom
merage:
    id1   val_x  id2   val_y
0  1.0   Jerry    1   Kerry
1  2.0     Tom    2   Peter
2  3.0   Kerry    3   Jerry
3  4.0  Jessca    4     Tom
4  NaN     NaN    5  Jessca
'''
def join_data(): #合并数组
    df1 = pd.DataFrame([['苏州','南京'],['广州','深圳'],['宁波','杭州']],
                       index =['江苏','广东','浙江'],
                       columns=['城市1','城市2'])                       
    df2 = pd.DataFrame([['赣州','南昌'],['成都','泸州'],['宁波','杭州']],
                       index =['江西','四川','浙江'],
                       columns=['城市3','城市4'])
    print("df1 is:\n",df1)
    print("df2 is:\n",df2)
    print("join:\n",df1.join(df2))
    print("outer join:\n",df1.join(df2,how='outer'))
'''
df1 is:
      城市1 城市2
江苏  苏州  南京
广东  广州  深圳
浙江  宁波  杭州
df2 is:
      城市3 城市4
江西  赣州  南昌
四川  成都  泸州
浙江  宁波  杭州
join:
      城市1 城市2  城市3  城市4
江苏  苏州  南京  NaN  NaN
广东  广州  深圳  NaN  NaN
浙江  宁波  杭州  宁波 杭州
outer join:
       城市1  城市2  城市3  城市4
四川   NaN    NaN    成都   泸州
广东   广州   深圳   NaN    NaN
江苏   苏州   南京   NaN    NaN
江西   NaN    NaN    赣州   南昌
浙江   宁波   杭州   宁波   杭州
'''
def concat_data():
    s1 = pd.Series([1,2],index=['苏州','南京'])
    s2 = pd.Series([3,4],index=['广州','深圳'])
    s3 = pd.Series([5,6],index=['宁波','杭州'])
    print("concat:\n",pd.concat([s1,s2,s3]))
    print("concat use key:\n",pd.concat([s1,s2,s3],keys=['江苏','广东','浙江']))
    df1 = pd.DataFrame([['苏州','南京'],['广州','深圳'],['宁波','杭州']],
                       index =['江苏','广东','浙江'],
                       columns=['城市1','城市2'])                       
    df2 = pd.DataFrame([['赣州','南昌'],['成都','泸州'],['宁波','杭州']],
                       index =['江西','四川','浙江'],
                       columns=['城市1','城市2'])
    print("concat Dataframe:\n",pd.concat([df1,df2],axis=0,keys=['数据1','数据2']))
    print("concat Dataframe:\n",pd.concat([df1,df2],axis=1,keys=['数据1','数据2']))
'''
concat:
苏州    1
南京    2
广州    3
深圳    4
宁波    5
杭州    6
dtype: int64
concat use key:
江苏  苏州    1
      南京    2
广东  广州    3
      深圳    4
浙江  宁波    5
      杭州    6
dtype: int64
concat Dataframe:
        城市1 城市2
数据1 江苏  苏州  南京
      广东  广州  深圳
      浙江  宁波  杭州
数据2 江西  赣州  南昌
      四川  成都  泸州
      浙江  宁波  杭州
concat Dataframe:
       数据1         数据2     
       城市1  城市2  城市1  城市2
江苏   苏州   南京  NaN     NaN
广东   广州   深圳  NaN     NaN
浙江   宁波   杭州  宁波   杭州
江西   NaN    NaN   赣州   南昌
四川   NaN    NaN   成都   泸州
'''
#2.3重塑和轴向旋转
def pivot_data():
   df = pd.DataFrame([{"编号":"1","姓名":"Jerry","科目":"语文","成绩":"92"},
                       {"编号":"1","姓名":"Jerry","科目":"数学","成绩":"97"},
                       {"编号":"2","姓名":"Tom","科目":"语文","成绩":"100"},
                       {"编号":"2","姓名":"Tom","科目":"数学","成绩":"91"}])
   print("df is:\n",df)
   print("df stack is:\n",df.stack())
   print("df unstack is:\n",df.unstack())
   print("df pivot is:\n",df.pivot('编号','科目','成绩'))
   print("set_index and unstack is:\n",df.set_index(['编号','科目']).unstack())
'''
df is:
   编号     姓名   科目   成绩
0  1        Jerry  语文   92
1  1        Jerry  数学   97
2  2        Tom    语文  100
3  2        Tom    数学   91
df stack is:
0  编号        1
   姓名       Jerry
   科目       语文
   成绩       92
1  编号        1
   姓名       Jerry
   科目       数学
   成绩       97
2  编号        2
   姓名       Tom
   科目       语文
   成绩      100
3  编号        2
   姓名       Tom
   科目       数学
   成绩       91
dtype: object
df unstack is:
编号0        1
    1        1
    2        2
    3        2
姓名0    Jerry
    1    Jerry
    2      Tom
    3      Tom
科目0       语文
    1       数学
    2       语文
    3       数学
成绩0       92
    1       97
    2      100
    3       91
dtype: object
df pivot is:
科目  数学   语文
编号         
1     97     92
2     91     100
set_index and unstack is:
        姓名         成绩     
科目     数学     语文  数学   语文
编号                       
1       Jerry  Jerry  97   92
2       Tom    Tom    91  100
'''
def melt_data():
    df = pd.DataFrame({'属性':['编号','年龄','手机'],
                      '张三':['1',54,'13681766555'],
                      '李四':['2',32,'13966564433'],
                      '王五':['3',48,'18977665643']
                    })
    print("df is:\n",df)
    melt_res = pd.melt(df,['属性'])
    print("melt:\n",melt_res)
    print("pivot:\n",melt_res.pivot('属性','variable','value'))
'''
df is:
    属性           张三           李四           王五
0  编号            1            2            3
1  年龄           54           32           48
2  手机  13681766555  13966564433  18977665643
melt:
    属性 variable        value
0  编号       张三            1
1  年龄       张三           54
2  手机       张三  13681766555
3  编号       李四            2
4  年龄       李四           32
5  手机       李四  13966564433
6  编号       王五            3
7  年龄       王五           48
8  手机       王五  18977665643
pivot:
 variable           张三           李四           王五
属性                                             
年龄                 54           32           48
手机        13681766555  13966564433  18977665643
编号                  1            2            3
'''

if __name__=="__main__":
    #data_info()
    #change_type()
    #judge_is_null()
    #replace_null()
    #is_duplicate()
    #drop_duplicate()
    #data_type_transfer()
    #data_replace()
    #data_categories()
    #data_filter()
    #str_deal()
    #data_index()
    #merge_data()
    #join_data()
    #concat_data()
    #pivot_data()
    melt_data()
