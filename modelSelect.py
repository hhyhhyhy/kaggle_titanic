#-*-coding:utf-8-*-
from kaggle_titanic_2 import  dataprocess
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingRegressor
from sklearn import linear_model

#模型打分
def scoreModel(estimator,X,y):
    score = estimator.oob_score_
    print("oob_score",score)#用out of bag数据来CV验证
    return score

def selectFeatureByModel(input_df):
    #特征列表 字符串名字
    features_list = input_df.columns.values[1::]
    X = input_df.values[:,1::]
    y = input_df.values[:,0]

    #我们如果将没有存活和存活下来的权重调整一下
    survived_weight = .75
    y_weights = np.array([survived_weight if s==0 else 1 for s in y])

    print("建立随机森林来看特征权重来选择特征")
    forest = RandomForestClassifier(oob_score=True,n_estimators=10000)
    forest.fit(X,y,sample_weight=y_weights)
    feature_importance = forest.feature_importances_
    #缩放到100以内
    feature_importance = 100.0*(feature_importance/feature_importance.max())
    # for i,feature in enumerate(fetures_list):
    #     print("feature:",feature,"the importance is:",feature_importance[i])
    #赛选出特征重要性并绘制图表

    fi_threshold = 18
    important_idx = np.where(feature_importance>fi_threshold)[0]#满足最小重要性的索引
    important_features = features_list[important_idx]
    print("\n", important_features.shape[0], "Important features(>", \
    fi_threshold, "% of max importance)...\n")
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]#倒排-1 满足条件的从大到小的索引
    #绘制重要性表格
    pos = np.arange(sorted_idx.shape[0])+.5
    plt.subplot(1,2,2)
    plt.title('Feature importance')
    plt.barh(pos,feature_importance[important_idx][sorted_idx[::-1]],color='r',align='center')
    plt.yticks(pos,important_features[sorted_idx[::-1]])
    plt.xlabel('Relative importance')
    plt.draw()
    plt.show()

    #注意根据样本数量适当调节特征个数
    #返回重要特征索引
    #X = X[:,important_idx][:,sorted_idx]#按什么顺序访问
    # submit_df = submit_df.iloc[:,important_idx].iloc[:,sorted_idx]#行列访问
    # print('\n训练所使用的特征大小', X.shape[1], "特征分别是:\n", submit_df.columns.values)

    return important_idx,sorted_idx



#超参数的选择
def turn_Random_forest_parameters(X,y,y_weights):
    #超参数的选择
    # criterion: 划分的规则，默认是gini。“gini” = Gini
    # Impurity，取值在0 - 1
    # 之间。“entropy” = 信息增益（information
    # gain）。基尼系数通常是确定平衡的一个指数，用于评价一个国家的收入是否分配不均衡。这里的基尼不纯度基本上恰好相反：值最小，=0，表明分类之后的元素都归于某一类，越纯（实际上对应的基尼系数应该是越不平衡）；越趋近于1，表明元素均匀的分散到各个分类里面。
    # splitter：划分节点的策略，默认是best，算法会根据criterion来选择最好的feature做分割。可以设置random，算法会随机选择feature做分割；但是实际上，也并非完全的随机，算法也会做一些避免造成泛化能力丢失的处理。
    # max_features: 划分的时候需要考虑多少特征，或者全部（默认值）或者一个子集。
    # max_depth: 最大树深度。避免过拟合的。
    # min_samples_split: 内部节点上，每一个节点至少需要有的sample个数。避免过拟合的。
    # min_samples_leaf:  叶子节点上，每一个节点至少需要有的sample个数。避免过拟合的。
    # min_weight_fraction_leaf: 没研究。
    # max_leaf_nodes: 最大叶子节点个数。他和max_depth互斥。避免过拟合的。
    # class_weight:分类的权重。没研究。
    # random_state: 随机种子，为splitter服务的。如果splitter = random，那么在对同一组数据做两次预测的时候，
    sqrtfeat = int(np.sqrt(X.shape[1]))
    minsampsplit = int(X.shape[0]*0.015)
    params_score = {"n_estimators":10000,
                    "max_features":sqrtfeat,#bag时候随机特征个数
                    "min_samples_split":minsampsplit}#最小的划分
    params = params_score
    print("Generating RandomForestClassifier model with parameters: ", params)
    forest = RandomForestClassifier(n_jobs=-1,oob_score=True,**params)
    #算每次的oob_Score
    test_scores = []
    # Using the optimal parameters, predict the survival of the labeled test set 10 times
    for i in range(5):
        forest.fit(X, y, sample_weight=y_weights)
        print("OOB:", forest.oob_score_)
        test_scores.append(forest.oob_score_)
    oob = np.mean(test_scores)
    print("oob mean:%.3f"% oob )
    print("分类器准确将袋外样本分类正确的个数:",np.mean(test_scores)*X.shape[0])
    return params




#绘制学习曲线，以确定模型的状况是否过拟合和欠拟合
def plot_learning_curve(estimator,title, X, y,ylim=(0.8, 1.01), cv=None,
                        train_sizes=np.linspace(.05, 0.2, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    """
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()



#若使用的是基分类器可以考虑模型融合来达到随机森林的效果

def BaggingModel(X,y,test_x):

    clf = linear_model.LogisticRegression(C=1.0,penalty="l1",tol=1e-6)
    bagging_clf = BaggingRegressor(clf,n_estimators=10,max_samples=0.8,max_features=1.0,
                                   bootstrap=True,bootstrap_features=False,n_jobs=-1)#是否放回bootstrap
    bagging_clf.fit(X,y)
    predictions = bagging_clf.predict(test_x)
    return predictions

def modelPredict():
    starttime = time.time()
    # 载入处理过的数据
    print("开始载入数据并特征工程------")
    input_df, submit_df = dataprocess.dataprocess()
    #删除人的序号
    #以及提交时候保留的编号属性
    drop_list = ['PassengerId']
    input_df.drop(drop_list,axis=1,inplace=1)
    submit_ids = submit_df['PassengerId']
    submit_df.drop(drop_list,axis=1,inplace=1)

    #选择重要特征
    important_idx,sorted_idx = selectFeatureByModel(input_df)

    X = input_df.values[:, 1::]
    y = input_df.values[:, 0]


    survived_weight = .75
    y_weights = np.array([survived_weight if s==0 else 1 for s in y])

    X = X[:,important_idx][:,sorted_idx]#按什么顺序访问
    test_df = submit_df.iloc[:,important_idx].iloc[:,sorted_idx]#行列访问
    print('\n训练所使用的特征大小', X.shape[1], "特征分别是:\n", test_df.columns.values)

    #分类器
    params = turn_Random_forest_parameters(X,y,y_weights)
    forest = RandomForestClassifier(n_jobs=-1, oob_score=True, **params)
    #通过学习曲线观测是否过拟合或者欠拟合来确定是否继续使用该模型
    #以及是否需要增加特征数量
    plot_learning_curve(LinearSVC(C=10.0), "LinearSVC(C=10.0)",
                        X, y, ylim=(0.3, 0.9),
                        train_sizes=np.linspace(.05, 0.2, 5))

    forest.fit(X,y)
    #提交
    print("开始预测测试样本并提交------")
    #转换成array类型
    submission =forest.predict(test_df).astype(int) #np.asarray(zip(submit_ids,forest.predict(submit_df))).astype(int)


    result = pd.DataFrame({'PassengerId': submit_ids.as_matrix(), 'Survived': submission})

    result.to_csv('randomforest_result.csv')


    print("Finish Successfuly!!\n")
    print("Work used Time %f",time.time()-starttime)

