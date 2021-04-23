import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import plot_roc_curve, auc
from sklearn.cluster import KMeans


#データの中にnullがあるかどうか調べる。
#in:   DataFrame
#out:  Dataframe(nullのあるcolumnNM, count)
def chkDfIsNull(df):
    chk_tmp = df.isnull().sum() #checking for total null values
    ISNULL = None
    for i in range(len(chk_tmp)):
        # extract null columns
        if chk_tmp[i] != 0:
            print("Column: " + str(df.columns[i]) + "   number: " + str(chk_tmp[i]))
            if ISNULL is None:
                ISNULL = pd.DataFrame([[df.columns[0], chk_tmp[0]]], columns = ["dfColumnNm", "null"])
            else:
                tmp = pd.DataFrame([[df.columns[i], chk_tmp[i]]], columns = ["dfColumnNm", "null"])
                ISNULL = ISNULL.append(tmp)
            
    if ISNULL is not None:
        ISNULL = ISNULL.reset_index(drop=True)
        return ISNULL
    
    print("There is not NULL.")
    return None


#指定列の一意性の確認
#in   :dataframe, column name
#out  :stats(if unique, return None.)
def chkUnique(df, clmnNm):
    numUniq = df[clmnNm].nunique()
    numObs = df.shape[0]
    if numObs == numUniq:
        print(clmnNm + " is unique.")
        return None
    else:
        print(clmnNm + " is not unique.")
        df = pd.DataFrame(df[clmnNm].value_counts()).reset_index()
        df.columns = ["uni_" + clmnNm, "nunique"]
        return df

    
#dfの相関係数を計算。
#絶対値として算出している。
def cnptCorr(df):
    #相関係数の導出 & series化　(相関係数の高い順にソートされている)
    correlations = df.iloc[:,1:].corr().abs().unstack().sort_values(kind="quicksort",ascending=False).reset_index()
    #同じ特徴同士の相関係数は排除
    correlations = correlations[correlations['level_0'] != correlations['level_1']].reset_index(drop=True)
    #列名をわかりやすく
    correlations.columns = ['level_0', 'level_1', "corr"]
    
    return correlations


#Adversarial Validation
#入力モデルのfitting、AUCの算出、ROCの描画を行う。
#In: 分類器のオブジェクト、CVオブジェクト(KFoldなど)、Feature、target(dataset_label)
def adv_roc(estimators, cv, X, y):

    fig, axes = plt.subplots(math.ceil(len(estimators) / 2), #ceil：整数へ切り上げ
                             2,
                             figsize=(16, 6))
    axes = axes.flatten()
    
    #分類器の数だけ回る
    for ax, estimator in zip(axes, estimators):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        #Kfoldで分けた分だけ回る(デフォルトは3)
        for i, (train, test) in enumerate(cv.split(X, y)):
            #学習
            estimator.fit(X.loc[train], y.loc[train])
            #学習済みモデルとテストセットよりROCを評価。戻り値は予測結果とグラフのオブジェクト。
            viz = plot_roc_curve(estimator,
                                 X.loc[test],
                                 y.loc[test],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3,
                                 lw=1,
                                 ax=ax)
            #０～１を100分割した空間への点を補完する。
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)  #y成分
            aucs.append(viz.roc_auc) #AUC

        ax.plot([0, 1], [0, 1],
                linestyle='--',
                lw=2,
                color='r',
                label='Chance',
                alpha=.8)

        #CV全体の平均と分散算出。
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr,
                mean_tpr,
                color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' %
                (mean_auc, std_auc),
                lw=2,
                alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr,
                        tprs_lower,
                        tprs_upper,
                        color='grey',
                        alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.02, 1.02],
               ylim=[-0.02, 1.02],
               title=f'{estimator.__class__.__name__} ROC for Adversarial Val.')
        ax.legend(loc='lower right', prop={'size': 10})
    plt.show()
    return



#Train,Testデータを一緒にしてKmeansにfitさせ、それぞれのデータにクラスターラベルを付与する。
#Input
#Train, test:データ。
#features:cluster分析対象のcolumn名の配列。
#n_clusters: クラスターの数。1クラスターにつき20レコードが割り振られるのが望ましい。
#kind:クラスタラベルのcolumn名(デフォルトは"clusterLabel")。
#seed:デフォルトは0。
#Output
#クラスタラベル付きのTrain,Test Data
def createClusterKmeans(train, test, features, n_clusters, kind = 'clusterLabel', seed = 0):
    train_ = train[features].copy()
    test_ = test[features].copy()
    data = pd.concat([train_, test_], axis = 0)
    kmeans = KMeans(n_clusters = n_clusters, random_state = seed).fit(data)
    train[f'{kind}'] = kmeans.labels_[:train.shape[0]]
    test[f'{kind}'] = kmeans.labels_[train.shape[0]:]
    return train, test