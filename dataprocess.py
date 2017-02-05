#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#一个属性一个属性的探索以及处理
#通过explore探测到具体的数值属性以及缺省值属性以及字符属性
#题意上给出了such as women, children, and the upper-class 也就是等级，年龄，性别非常相关
#同时探索到std变换较大的是age和Fare属性

#Pclass
#explore后可能等级越高获救的可能性越大
#但是不能确定是否一定所以我们用虚拟化来处理one-hot_encoder相当于
#最后剩下Pclass-scaled
def processPclass(df):
    df.Pclass[df.Pclass.isnull()] = df.Pclass.dropna().mode().values
    dummies_df = pd.get_dummies(df['Pclass'],prefix='Pclass')#前缀
    df = pd.concat([df,dummies_df],axis=1)
    scaler = preprocessing.StandardScaler()
    df['Pclass_scaled'] = scaler.fit_transform(df['Pclass'])
    return df

#Name属性
#提到的关键属性有年龄性别，等级 均有说明整个属性比较重要
#长度长说明这个人可能地位高
#字符类型可能需要离散化或者虚拟化
#由于属性需要取值种类太多。我们将所有替换为几个关键的属性

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring)!=-1:
            return substring
    return np.nan

#替换规则
#若只跟某几个名字有关那么我们需要虚拟化扩维
def replace_titles(x):
    title = x['Title']
    if title in ['Mr', 'Don', 'Major', 'Capt', 'Sir', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Jonkheer']:
        return 'Master'
    elif title in ['Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'male':
            return 'Mr'
        else:
            return 'Mrs'
    elif title == '':
        if x['Sex'] == 'male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title
from  sklearn  import preprocessing
le = preprocessing.LabelEncoder()
def processName(df):
    df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))  # 这个名字有几个词也就是长度
    df['Title'] = df['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])#其中第一个.是任意字符
    #df['Title'] = df.apply(replace_titles,axis=1)
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

    # group low-occuring, related titles together
    df['Title'][df.Title == 'Jonkheer'] = 'Master'
    df['Title'][df.Title.isin(['Ms', 'Mlle'])] = 'Miss'
    df['Title'][df.Title == 'Mme'] = 'Mrs'
    df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
    df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

    #虚拟化只跟某几种有关
    df = pd.concat([df,pd.get_dummies(df['Title']).rename(columns=lambda x:'Title_'+str(x))],axis=1)

    # #字符类型离散化
    # le.fit(df['Title'])
    # title_le = le.transform(df['Title'])
    # df['Title_id'] = title_le.astype(np.int32)
    df['Title_id'] = pd.factorize(df['Title'])[0]+1

    #标准化是为了后续组合特征以及PCA等
    scaler = preprocessing.StandardScaler()
    df['Names_scaled'] = scaler.fit_transform(df['Names'])
    scaler.fit(df['Title_id'])
    df['Title_id_scaled'] = scaler.transform(df['Title_id'])
    return df


#Sex属性
#探索后发现女性存活率高大概百分之75左右所以是关键属性
#文本-》数值 由于男女有区别所有转换成数值就行了
def processSex(df):
    df['Gender'] = df['Sex'].map({'female':0,'male':1}).astype(np.int32)
    return df


#Age属性
#std变换较大。缺省值有
#20-30最多人然后小孩第二多。最少的是老人
#小孩获救以及老人获救几率较高。其他差不多一般左右

def setMissingData(df,features=[],missFeature='Age'):
    feature_df = df[features]
    X = feature_df[df[missFeature].notnull()].as_matrix()[:,1::]
    y = feature_df[df[missFeature].notnull()].as_matrix()[:,0]
    rtr = RandomForestRegressor(n_estimators=2000,n_jobs=-1)#无限制处理机
    rtr.fit(X,y)
    predicitedAges = rtr.predict(feature_df[df[missFeature].isnull()].as_matrix()[:,1:])
    df.loc[(df[missFeature].isnull()),missFeature] = predicitedAges
    return df

# def setMissingAges(df):
#
#     age_df = df[['Age', 'Embarked', 'Fare', 'Parch', 'SibSp', 'Title_id', 'Pclass', 'Names', 'CabinLetter']]
#     X = age_df.loc[(df.Age.notnull())].values[:, 1::]
#     y = age_df.loc[(df.Age.notnull())].values[:, 0]
#
#     rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
#     rtr.fit(X, y)
#
#     predictedAges = rtr.predict(age_df.loc[(df.Age.isnull())].values[:, 1::])
#     df.loc[(df.Age.isnull()), 'Age'] = predictedAges
#     return  df
def processAge(df):
    #先填缺省值
    #预测的方法RandomForest
    df = setMissingData(df, features=['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title_id','Pclass','Names','CabinLetter'], missFeature='Age')
    #df = setMissingAges(df)
    #此处用中位数以及均值填充但是需要先分层再求均值。
    # mean_master = np.average(df['Age'][df.Title=='Master'].dropna())
    # mean_mr = np.average(df['Age'][df.Title=='Mr'].dropna())
    # mean_miss = np.average(df['Age'][df.Title=='Miss'].dropna())
    # mean_mrs = np.average(df['Age'][df.Title=='Mrs'].dropna())
    # df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age'] = mean_master
    # df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'Age'] = mean_mr
    # df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age'] = mean_miss
    # df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'Age'] = mean_mrs
    scaler = preprocessing.StandardScaler()
    df['Age_scaled'] = scaler.fit_transform(df['Age'])
    #特别提到老人小孩。那么显然要离散化年龄
    # bin into quartiles and create binary features
    #按照频率接近的类别编号在一起
    df['Age_bin'] = pd.qcut(df['Age'],4)
    #而若只跟几个年龄段有关跟其他无关那么虚拟化要
    df = pd.concat([df, pd.get_dummies(df['Age_bin']).rename(columns=lambda x: 'Age_' + str(x))], axis=1)

    df['Age_bin_id'] = pd.factorize(df['Age_bin'])[0]+1
    #Age_bin_id也要标准化为了后续组合以及PCA方便
    scaler = preprocessing.StandardScaler()
    df['Age_bin_id_scaled'] = scaler.fit_transform(df['Age_bin_id'])
    df['Child'] = (df['Age']<13).astype(int)

    #变化不大
    # from sklearn import  preprocessing
    # scaler = preprocessing.StandardScaler()
    # df['Age_bin_id_scaled'] = scaler.fit_transform(df['Age_bin_id'])
    return  df


#处理兄弟姐妹和配偶的数量
#联想到如果这个数目较多的话可能是兄弟看年龄较小的话就是孩子
#而这两个属性我们联想到其实可以融合成一个家庭属性。而且人数越多情况有可能没获救
#0比较多我们算上自己
def processSibsp(df):
    df['SibSp'] = df['SibSp'] + 1#也为了能够标准化
    scaler = preprocessing.StandardScaler()
    df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'])
    #有可能只跟特别的几个有关
    sibsps = pd.get_dummies(df['SibSp']).rename(columns=lambda x: 'SibSp_' + str(x))
    parchs = pd.get_dummies(df['Parch']).rename(columns=lambda x: 'Parch_' + str(x))
    df = pd.concat([df, sibsps, parchs], axis=1)
    return df

def processParch(df):
    df['Parch'] = df['Parch'] + 1
    scaler = preprocessing.StandardScaler()
    df['Parch_scaled'] = scaler.fit_transform(df['Parch'])
    return df

def processFamily(df):
    df = processSibsp(df)
    df = processParch(df)
    #df['Family'] = df['SibSp'] + df['Parch']
    return df

#处理Fare std变换较大Scaled必须的。同时0比较多平滑化.而和关键属性猜想和upper class有关
#活下来的人，Fare总花费较多
#算一下每个人的花费
# 的可能性比较大可能人物越有
#最后有用的属性Fare_scaled Fare_bin_id_scaled
def processFare(df):
    #std处理方式有很多。最基本是scaled.还可以离散化
    #先平滑化使得0没有那么多
    df['Fare'][df.Fare.isnull()] = 0 #
    df.loc[(df.Fare==0),'Fare'] = df['Fare'][df.Fare.nonzero()[0]].min()/10

    sclar = preprocessing.StandardScaler()
    df['Fare_scaled'] = sclar.fit_transform(df['Fare'])
    df['Fare_bin'] = pd.qcut(df['Fare'],4)
    df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)#只跟几个段有关的时候
    df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0] + 1
    #为了后续步骤
    df['Fare_bin_id_scaled'] = sclar.fit_transform(df['Fare_bin_id'])
    return df

#我们猜测和等级有关。类似名字处理方式无法离散化。只有用几个代表的类型来表示
#我们需要分析票的前缀同时发现和Pclass的关系也就是说前缀确实有等级的关系
#这种字符串我们都需要剖析前缀和后缀然后编码
#获取前缀
import re
def getTicketPrefix(ticket):
    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:#有前缀
        return match.group()
    else:#没有前缀
        return 'U'

#获取票的数字
def getTicketNumber(ticket):
    match = re.compile("([0-9]+$)").search(ticket)
    if match:
        return  match.group()
    else:
        return '0'

#处理后有用的属性就Prefix Number_scaled Length,Start
def processTicket(df):
    #先化作字符串数组
    #strTick = df.Ticket.map(lambda x: str(x))
    #lenTick = df.Ticket.map(lambda x: len(x))
    df['TicketPrefix'] = df['Ticket'].map(lambda x: getTicketPrefix(x.upper()))
    df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('[\.?\/?]', '', x))  # 去掉./
    df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('STON', 'SOTON', x))
    df['TicketPrefix'] = pd.factorize(df['TicketPrefix'])[0]
    df['TicketNumber'] = df['Ticket'].map(lambda x: getTicketNumber(x))
    df['TicketNumberLength'] = df['TicketNumber'].map(lambda x: len(x)). astype(int)
    df['TicketNumberStart'] = df['TicketNumber'].map(lambda x: x[0:1]).astype(int)
    df['TicketNumber'] = df['TicketNumber'].astype(int)
    #有可能跟票号有关系但是这个数显然不是标准属性
    scaler = preprocessing.StandardScaler()
    df['TicketNumber_scaled'] = scaler.fit_transform(df['TicketNumber'])
    return df


#Cabin有很多缺省值
#发现是否缺省对结果影响很大
#同时参数贡献度不够的说明划分不够细
#能代表着一些隐含信息。比如船舱号Cabin这一属性，缺失可能代表并没有船舱
#发现字母跟等级有关
def getCabinLetter(cabin):
    match = re.compile("([a-zA-Z]+)").search(cabin)
    if match:
        return  match.group()
    else:
        return 'U'

def getCabinNumber(cabin):
    match = re.compile("([0-9]+)").search(cabin)
    if match:
        return match.group()
    else:
        return 0

#最后有用的属性就只有CabinLetter 和CabinNumber_scaled
def processCabin(df):
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    df['CabinLetter'] = df['Cabin'].map(lambda x:getCabinLetter(x))
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
    df['CabinNumber'] = df['Cabin'].map(lambda x:getCabinNumber(x)).astype(int) + 1 #0太多
    #std比较大所以我们要标准化
    sclar = preprocessing.StandardScaler()
    df['CabinNumber_scaled'] = sclar.fit_transform(df['CabinNumber'])
    return df



#处理Embarked 上船位置
#猜想上船的位置靠近救生处以及是以最近等级高的地方所以这个属性需要
#S窗口最多。C获救几率最大
#最后剩下Embarked_  Emarked_id
def processEmbarked(df):
    df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode()#纵数
    # #如果根据数字变化有一定趋势。
    # df.Embarked = pd.factorize(df['Embarked'])[0]#数值化
    # #也可以虚拟化
    # 如果只跟其中一个有关而与其他无关的时候那么可以
    df = pd.concat([df,pd.get_dummies(df['Embarked'],prefix='Embarked')],axis=1)
    df.Embarked = pd.factorize(df['Embarked'])[0]
    return df



#考虑组合特征
#而我们组合特征其实是把每一维特征权重应该相等看待。所以最好都scaled
#然后我们遍历所有可能的组合
#这些特征都标准化过说明一样重要
def combineFeature(df):
    print("Starting With",df.columns.size,"手动生成组合特征",df.columns.values)
    #只考虑连续属性同时标准化过的属性
    numerics = df.loc[:,['Age_scaled','Fare_scaled','Pclass_scaled','Parch_scaled','SibSp_scaled',
                         'Names_scaled','CabinNumber_scaled','Age_bin_id_scaled','Fare_bin_id_scaled']]
    print("\nFeatures used for automated feature generation:\n",numerics.head(10))

    new_fields_count = 0
    for i in range(0,numerics.columns.size - 1):
        for j in range(0,numerics.columns.size-1):
            if i<=j:
                name = str(numerics.columns.values[i]) + "*" + str(numerics.columns.values[j])
                df = pd.concat([df,pd.Series(numerics.iloc[:,i]*numerics.iloc[:,j],name=name)],axis=1)
                new_fields_count+=1
            if i<j:
                name = str(numerics.columns.values[i]) + "+" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] + numerics.iloc[:, j], name=name)], axis=1)
                new_fields_count += 1
            if not i == j:
                name = str(numerics.columns.values[i]) + "/" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] / numerics.iloc[:, j], name=name)], axis=1)
                name = str(numerics.columns.values[i]) + "-" + str(numerics.columns.values[j])
                df = pd.concat([df, pd.Series(numerics.iloc[:, i] - numerics.iloc[:, j], name=name)], axis=1)
                new_fields_count += 2
    print("\n",new_fields_count,"new features generated")
    return df

#数据规约和降低维度
# 数据规约维度 PCA+ KMeans
# 但仍接近于保持原始数据的完整性。也就是说，在归约后的数据集上进行数据挖掘将更加有效，仍然产生几乎相同的数据分析结果。
# PCA通过创建一个替换的较小的变量集组合属性的基本要素
# 规约有属性和数值

#特征提取降低维度
def processDrops(df):
    #手动过程
    #删除被替换的属性我们一个属性一个属性的排除 std大的以及字符的都要删除
    #而一些标准化过的属性会在后续协方差中删除
    #可以看作删除的是最初给定的一些属性
    #rawDropList = ['Name', 'Names', 'Title', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked', \
    #               'Cabin', 'CabinLetter', 'CabinNumber', 'Age', 'Fare', 'Ticket', 'TicketNumber','Age_bin','Fare_bin']
    #CabinNumber删除因为CabinNumber_scaled应该留下
    DropList = ['Cabin','CabinNumber','Ticket','TicketNumber','Fare','Fare_bin','Embarked',\
                 'Parch','SibSp','Name','Title','Sex','Pclass','Age','Age_bin']
    #删除字符属性的值
    #stringsDropList = ['Title', 'Name', 'Cabin', 'Ticket', 'Sex', 'Ticket', 'TicketNumber','Age_bin','Fare_bin']
    df.drop(DropList,axis=1,inplace=True)

    #发现排除survived 和passengerId以后的子列
    df_corr = df.drop(['Survived', 'PassengerId'], axis=1).corr(method='spearman')
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)#默认中间为0相当于跟自己无关
    df_corr = mask * df_corr#乘是对应元素乘
    #得到整体的协方差矩阵

    #赛选出相关性太高的属性冗杂属性
    drops = []
    for col in df_corr.columns.values:
        if np.in1d([col],drops):#已经有了
            continue
        corr = df_corr[abs(df_corr[col]) > 0.98].index#第col列中相关性太大的舍弃
        drops = np.union1d(drops, corr)
    print("\nDropping", drops.shape[0], "highly correlated features...\n")# , drops
    df.drop(drops,axis=1,inplace=True)


    return  df

#我们得到了具有大量特征的维度很高的数据集，特征较多不能直接用来作为模型输入，一是因为这些特征间具有多重共线性，
# 可能 会导致空间的不稳定；二是因为高维空间本身具有稀疏性，一维正态分布有68%的值落于正负标准差之间，而在十维空间上只有0.02%；三是由于过多的属性
# 会使挖掘需要很长时间。对于一些模型来说，比如使用L1（Lasso），当有大量属性时效果很好，因为它可以有效忽略掉噪声变量。而一些模型则容易过拟 合。
#会使用聚类分析和降维来处理数据
#归约步骤
def reduceAndCluster(df,input_df,clusters=3):
    #属性规约
    # #删除人的序号
    passengerIds = df['PassengerId']
    drop_list = ['PassengerId']
    df.drop(drop_list,axis=1,inplace=1)
    survivedSeries = pd.Series(df['Survived'],name='Survived')#将值拿出来
    #df.drop('Survived',axis=1,inplace=True)
    #df = df.reindex_axis(input_df.columns,axis=1)#重新按照input_df列方式
    #print(df.head())
    X = df.values[:,1::]
    y = df.values[:, 0]#类别属性

    #print(X[:5])
    variance_pct = .99

    # Create PCA object
    pca = PCA(n_components=variance_pct)

    # Transform the initial features
    X_transformed = pca.fit_transform(X,y)

    # Create a data frame from the PCA'd data
    pcaDataFrame = pd.DataFrame(X_transformed)

    print("原数据维度",X.shape[1])
    print("PCA后维度",X_transformed.shape[1])


    #值规约。聚类
    from sklearn.cluster import KMeans

    #聚类属性 探测样本间的相关性
    kmeans = KMeans(n_clusters=clusters, random_state=np.random.RandomState(4), init='random')

    #我们分为训练集和测试集
    #split_train, split_cv = train_test_split(df, test_size=0.2, random_state=0)

    trainClusterIds = kmeans.fit_predict(X_transformed[:input_df.shape[0]])#得到每个样本的聚类中心
    print("训练样本的聚类中心是:",trainClusterIds)

    testClusterIds = kmeans.predict(X_transformed[input_df.shape[0]:])
    # print
    # "clusterIds shape for test data: ", testClusterIds.shape
    print ("测试样本的聚类中心是: ", testClusterIds)

    clusterIds = np.concatenate([trainClusterIds,testClusterIds])
    print("整体的样本中心: ", clusterIds.shape)

    # 创建聚类中心的Id
    clusterIdSeries = pd.Series(clusterIds, name='ClusterId')
    df = pd.concat([survivedSeries, clusterIdSeries, pcaDataFrame], axis=1)

    df = pd.concat([passengerIds,df],axis=1)
    return df



#数据清理以及特征工程
def clean_and_feature_engineer(df,input_df,is_pca=False):
    df = processCabin(df)
    df = processEmbarked(df)
    df = processTicket(df)
    df = processFare(df)
    df = processFamily(df)
    df = processName(df)
    df = processSex(df)
    df = processPclass(df)
    df = processAge(df)
    #组合特征
    df = combineFeature(df)

    #删除冗杂属性
    df = processDrops(df)

    #PCA降维的化才进行规约

    if is_pca:
        df = reduceAndCluster(df,input_df)
    return df


# #主函数
def dataprocess(filename='train.csv',testname='test.csv',is_pca=False):
#     #将训练和测试组合
#重新索引以及去掉NAN的survived提交变量
    # 将训练和测试组合
    input_df = pd.read_csv(filename)
    submit_df = pd.read_csv(testname)
    df = pd.concat([input_df, submit_df])
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    df = clean_and_feature_engineer(df,input_df,is_pca)

    #df = reduceAndCluster(df,input_df)
    input_df = df[:input_df.shape[0]]
    submit_df = df[input_df.shape[0]:]
    submit_df.reset_index(inplace=True)
    submit_df.drop('index', axis=1, inplace=True)
    submit_df.drop('Survived', axis=1, inplace=1)
    return input_df,submit_df
