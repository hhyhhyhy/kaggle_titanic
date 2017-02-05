#-*-coding:utf-8-*-
import numpy as np
import pandas as pd

df = pd.read_csv('train.csv')

#缺省值属性Age Cabin Embarked
# df.info()
#
# df.explore()


def set_ch():
	from pylab import mpl
	mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
	mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

set_ch()
#PcLass属性
# import matplotlib.pyplot as plt
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
# survived_0 = df.Pclass[df.Survived==0].value_counts()
# survived_1 = df.Pclass[df.Survived==1].value_counts()
# data_frame = pd.DataFrame({u"获救":survived_0,u"未获救":survived_1})
# data_frame.plot(kind='bar')
# plt.show()

#Name属性
#Miss获救比较多
# import matplotlib.pyplot as plt
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
# Master_survived = df.Survived[df.Name.str.contains("Master.")].value_counts()
# Mr_survived = df.Survived[df.Name.str.contains("Major.")].value_counts()
# data_frame = pd.DataFrame({u"1":Master_survived,u"2":Mr_survived})
# data_frame.plot(kind='bar')
# plt.show()
#
#
# #sex属性，文本-》数值 由于男女有区别所有转换成数值就行了
# #sex 探索
# survived_0 = df.Sex[df.Survived==0].value_counts()
# survived_1 = df.Sex[df.Survived==1].value_counts()
# data_frame = pd.DataFrame({u"获救":survived_0,u"未获救":survived_1})
# #黄色是female
# #data_frame.plot(kind='bar')
# female_1 = df[df.Survived==0][df.Sex=='female']
# female_1 = female_1.shape[0]
# female_1
# female_2 = df[df.Survived==1][df.Sex=='female'].shape[0]
# sizes = [female_1,female_2]
# plt.pie(sizes,shadow=True,autopct='%1.1f%%')
# plt.show()
# from sklearn.ensemble import RandomForestRegressor
# def setMissingData(df,features=[],missFeature='Age'):
#     feature_df = df[features]
#     X = feature_df[df[missFeature].notnull()].as_matrix()[:,1:]
#     y = feature_df[df[missFeature].notnull()].as_matrix()[:,0]
#     rtr = RandomForestRegressor(n_estimators=2000,n_jobs=-1)#无限制处理机
#     rtr.fit(X,y)
#     predicitedAges = rtr.predict(feature_df[df[missFeature].isnull()].as_matrix()[:,1:])
#     df.loc[(df[missFeature].isnull()),missFeature] = predicitedAges
#     return df
# df = setMissingData(df,features=['Age','Fare', 'Parch', 'SibSp', 'Pclass'],missFeature='Age')
#



#Sibsp 分析出来是两种人。一种是有父母的亲兄弟年龄普遍小
#另一种是年老的
df[df.SibSp>1]



#分析和fare的关系
# plt.subplot(1,2,1)
# survived_1 = df.Fare[df.Survived==1]
# survived_1.plot(kind='hist')
# plt.subplot(1,2,2)
# survived_0 = df.Fare[df.Survived==0]
# survived_0.plot(kind='hist')
# plt.show()


import re
def getTicketPrefix(ticket):
    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:#有前缀
        return match.group()
    else:#没有前缀
        return 'U'

def processTicket(df):
    #先化作字符串数组
    #strTick = df.Ticket.map(lambda x: str(x))
    #lenTick = df.Ticket.map(lambda x: len(x))
    df['TicketPrefix'] = df.Ticket.map(lambda x:getTicketPrefix(x.upper()))
    df['LenTick'] = df.Ticket.map(lambda x: len(x))

df = processTicket(df)
