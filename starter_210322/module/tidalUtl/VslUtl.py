import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import norm
from sklearn.decomposition import PCA
import plotly.express as px

cust_palt = ['#111d5e','#c70039','#37b448','#B43757', '#ffbd69', '#ffc93c','#FFFF33','#FFFACD',]

#指定列のカテゴリカルな特徴量のヒストグラムをtrain, testの順に並べる。
#in  :train data, testdata, visualizing column name
#out :-
def histCategory(dfTrain, clmnNm, dfTest=None):
    figure = plt.figure(figsize=(12, 4))
    gs_master = GridSpec(nrows=1, ncols=2, figure=figure)
    
    ax1 = figure.add_subplot(gs_master[0, 0])
    ax1.set_title("Train " + clmnNm,weight='bold')
    sns.countplot(x=clmnNm,
                  data=dfTrain,
                  ax=ax1,
                  order=dfTrain[clmnNm].value_counts().index)
    total = float(len(dfTrain[clmnNm]))
    for p in ax1.patches:
        ax1.text(p.get_x() + p.get_width() / 2., #width(x) :get_xは棒グラフの左端の位置
                p.get_height() + 2, #height(y)
                '{:1.2f}%'.format((p.get_height() / total) * 100),
                ha='center')
    
    
    if dfTest is not None:
        ax2 = figure.add_subplot(gs_master[0, 1])
        ax2.set_title("Test " + clmnNm,weight='bold')
        sns.countplot(x=clmnNm,
                      data=dfTest,
                      ax=ax2,
                      order=dfTest[clmnNm].value_counts().index)
        total = float(len(dfTest[clmnNm]))
        for p in ax2.patches:
            ax2.text(p.get_x() + p.get_width() / 2., #width(x) :get_xは棒グラフの左端の位置
                    p.get_height() + 2, #height(y)
                    '{:1.2f}%'.format((p.get_height() / total) * 100),
                    ha='center')
    
    return



# Index付き1次元データフレームのヒストグラム表現(横向き)
def histCntHorizontal(cntWithIdx):
    fig = plt.figure(figsize=(20,15))
    sns.barplot(y = cntWithIdx.reset_index()["index"].astype(str), x = cntWithIdx.values)
    plt.show()
    return



# Index付き1次元データフレームのヒストグラム表現(縦向き)
def histCntVertical(cntWithIdx):
    fig = plt.figure(figsize=(20,10))
    sns.barplot(x = cntWithIdx.reset_index()["index"].astype(str), y = cntWithIdx.values)
    plt.show()
    return



# 1次元データフレームのカーネル密度推定法(KDE)による分布表現
def distKde(distDf):
    fig = plt.figure(figsize=(20,10))
    sns.kdeplot(distDf.values.reshape(1, len(distDf))[0], shade=True)
    plt.show()
    return



#meta情報(mean, median, min, max, std, variance, skew(歪度), kurtosis(尖度))の分布
#train, testのメタ情報の差異を調べる
def metaDist(dfTrain, dfTest):
    fig = plt.figure(constrained_layout=True, figsize=(20, 16))
    grid = GridSpec(ncols=4, nrows=4, figure=fig)
    
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Distribution of Mean Values per Column', weight='bold')
    sns.kdeplot(dfTrain.mean(axis=0),color=cust_palt[0], shade=True, label='Train')
    sns.kdeplot(dfTest.mean(axis=0),color=cust_palt[1], shade=True, label='Test')
    
    ax2 = fig.add_subplot(grid[0, 2:])
    ax2.set_title('Distribution of Median Values per Column', weight='bold')
    sns.kdeplot(dfTrain.median(axis=0),color=cust_palt[0], shade=True, label='Train')
    sns.kdeplot(dfTest.median(axis=0),color=cust_palt[1], shade=True, label='Test')
    
    ax3 = fig.add_subplot(grid[1, :2])
    ax3.set_title('Distribution of Minimum Values per Column', weight='bold')
    sns.kdeplot(dfTrain.min(axis=0),color=cust_palt[0], shade=True, label='Train')
    sns.kdeplot(dfTest.min(axis=0),color=cust_palt[1], shade=True, label='Test')
    
    ax4 = fig.add_subplot(grid[1, 2:])
    ax4.set_title('Distribution of Maximum Values per Column', weight='bold')
    sns.kdeplot(dfTrain.max(axis=0),color=cust_palt[0], shade=True, label='Train')
    sns.kdeplot(dfTest.max(axis=0),color=cust_palt[1], shade=True, label='Test')
    
    ax5 = fig.add_subplot(grid[2, :2])
    ax5.set_title('Distribution of Std\'s per Column', weight='bold')
    sns.kdeplot(dfTrain.std(axis=0),color=cust_palt[0], shade=True, label='Train')
    sns.kdeplot(dfTest.std(axis=0),color=cust_palt[1], shade=True, label='Test')
    
    ax6 = fig.add_subplot(grid[2, 2:])
    ax6.set_title('Distribution of Variances per Column', weight='bold')
    sns.kdeplot(dfTrain.var(axis=0),color=cust_palt[0], shade=True, label='Train')
    sns.kdeplot(dfTest.var(axis=0),color=cust_palt[1], shade=True, label='Test')
    
    ax7 = fig.add_subplot(grid[3, :2])
    ax7.set_title('Distribution of Skew Values per Column', weight='bold')
    sns.kdeplot(dfTrain.skew(axis=0),color=cust_palt[0], shade=True, label='Train')
    sns.kdeplot(dfTest.skew(axis=0),color=cust_palt[1], shade=True, label='Test')
    
    ax8 = fig.add_subplot(grid[3, 2:])
    ax8.set_title('Distribution of Kurtosis Values per Column', weight='bold')
    sns.kdeplot(dfTrain.kurtosis(axis=0),color=cust_palt[0], shade=True, label='Train')
    sns.kdeplot(dfTest.kurtosis(axis=0),color=cust_palt[1], shade=True, label='Test')
    
    plt.suptitle('Meta Distributions of Train/Test Set', fontsize=25, weight='bold')
    plt.show()
    
    return



#dfのFeatureごとの分布図を表示。
#オプションでdf2(test data)も並べられる。
#rows×columnsをdfのfeature数に合わせること。
def featDist(df, cols, rows=3, columns=3, figsize=(30,25), title=None, dfOpt=None):
    
    fig, axes = plt.subplots(rows, columns, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for i, j in zip(cols, axes):
        sns.distplot(
                    df[i],
                    ax=j,
                    hist=False,
                    #color='#111d5e',
                    label=f'Train {i}',
                    kde_kws={'alpha':0.9})        
        
        if dfOpt is not None:
            sns.distplot(
                        dfOpt[i],
                        ax=j,
                        hist=False,
                        color = '#c70039',
                        label=f'Test {i}',
                        kde_kws={'alpha':0.7})
        
        j.set_title('Train Test Dist of {0}'.format(i.capitalize()), weight='bold')
        fig.suptitle(f'{title}', fontsize=24, weight='bold')

    return



#dfの分布図と最尤法でfittingした正規分布を同時に表示。差異を調べる。
#オプションでdf2(test data)も並べられる。
#rows×columnsをdfのfeature数に合わせること。
#範囲を設定する場合はdomain = [xmin, xmax, ymin, ymax] を設定すること
def featDistNorm(df, cols, rows=3, columns=3, figsize=(30,25), title=None, dfOpt=None, domain=None):
    
    fig, axes = plt.subplots(rows, columns, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for i, j in zip(cols, axes):
        sns.distplot(
                    df[i],
                    ax=j,
                    fit=norm,
                    hist=False,
                    #color='#111d5e',
                    label=f'Train {i}',
                    kde_kws={'alpha':0.9})        
        
        if dfOpt is not None:
            sns.distplot(
                        dfOpt[i],
                        ax=j,
                        hist=False,
                        color = '#c70039',
                        label=f'Test {i}',
                        kde_kws={'alpha':0.7})
        
        if domain is not None:
            j.axis(domain)
        
        (mu, sigma) = norm.fit(df[i])
        j.set_title('Train Test Dist of {0} Norm Fit: $\mu=${1:.2g}, $\sigma=${2:.2f}'.format(i.capitalize(), mu, sigma), weight='bold')
        fig.suptitle(f'{title}', fontsize=24, weight='bold')
        
    return


#相関関係のヒートマップを表示
#in: df.corr()で作られたデータフレーム
def corrHeatMap(correlation):
    mask = np.triu(correlation)
    plt.figure(figsize=(30, 12))
    sns.heatmap(correlation,
                mask=mask,
                annot=True,
                fmt='.3f',
                cmap='Wistia',
                linewidths=0.05,
                cbar=True)
    
    
    plt.title('Features with Highest Correlations',  weight='bold')
    plt.show()
    return



#PCAによるデータの次元削減
#PCA結果のパレート図
#in :dfTrain, dfTest, Rank:上位何位まで見せるか(barの数)
def parateResultPCA(dfTrain, dfTest, Rank=None):
    #主成分解析によるベクトル変換
    pca = PCA()
    pca.fit(dfTrain.iloc[:,1:])
    pcaTrain = pca.transform(dfTrain.iloc[:,1:])
    pcaTest = pca.transform(dfTest.iloc[:,1:])
    
    #パレート図
    fig, ax = plt.subplots(1,1,figsize=(30, 12))
    ax.plot(range(dfTrain.iloc[:,1:].shape[1]), pca.explained_variance_ratio_.cumsum(), linestyle='--',
               drawstyle='steps-mid', color=cust_palt[1], label='Cumulative Explained Variance')
    sns.barplot(np.arange(1,dfTrain.iloc[:,1:].shape[1]+1), pca.explained_variance_ratio_, alpha=0.85, color=cust_palt[0],
                label='Individual Explained Variance', ax=ax)
    ax.set_ylabel('Explained Variance Ratio', fontsize = 14)
    
    #範囲の制限([0,Rank-1,0,1])
    if Rank is None:
        ax.set_title('Explained Variance', fontsize = 20, weight='bold')
        ax.set_xlabel('Number of Principal Components', fontsize = 14)
        plt.legend(loc='center right', fontsize = 13);
        ax.set_xticks([])
    else:
        ax.axis([0,Rank-1,0,1])
        ax.set_title('First ' + str(Rank) + ' Explained Variances', fontsize = 20, weight='bold')
        ax.set_xlabel('Number of Principal Components', fontsize = 14)
    
    plt.tight_layout()
    
    return


#pca成分の組み合わせ事のグラフを表示(対応するClmnによって色付け)
#in :学習済みpcaモデル、pca変換後のtrain, trainの生データ, 色付けするcolumn名
#※※　グラフはpcaTrainの列数^2の行列で表示されるのであらかじめ次元削減しておくこと ※※
def distVarianceforClmn(pca, pcaTrain, Train, colorClmn):
    total_var = pca.explained_variance_ratio_.sum() * 100
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    
    fig = px.scatter_matrix(
        pcaTrain,
        color=Train.iloc[:,1:][colorClmn],
        dimensions=range(pcaTrain.shape[1]),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}% vs ' + colorClmn,
        opacity=0.5,
        color_discrete_sequence=cust_palt[:pcaTrain.shape[1]],
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()
    return