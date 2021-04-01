import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer



#ラベルエンコード(文字列→数値)
#feature_nameの例：['cp_time','cp_dose']
def Label_encode(train, test, feature_name):
    for f in feature_name:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))
    
    return train, test


#ラベルエンコード(One-Hot)
#feature_nameの例：['cp_time','cp_dose']
def OneHot_encode(train, test, feature_name):
    train = pd.get_dummies(train, columns=feature_name)
    test = pd.get_dummies(test, columns=feature_name)
    
    return train, test



#指定したcolumnと値に応じて欠損値補完を行い、新たにXXX_isnan列を設ける。
def FillnaAndInsertIsnan(DataFrame, ColsAndFillVals):
    dfIsNan = None
    for (col, val) in ColsAndFillVals:
        #欠損値の位置を示すbool列生成
        IsnanSeries = DataFrame[col].isnull()
        #欠損値を補完。
        DataFrame[col] = DataFrame[col].fillna(val)
        #Insert用の欠損値の位置を示すbool列
        if dfIsNan is None:
            dfIsNan = pd.DataFrame(IsnanSeries,columns=[IsnanSeries.name])
            dfIsNan = dfIsNan.rename(columns={IsnanSeries.name: IsnanSeries.name + "_isnan"})
        else:
            dfIsNan.insert(len(dfIsNan.columns), col + "_isnan", IsnanSeries)
        
    return DataFrame, dfIsNan



#主成分解析によるデータの次元削減
#in :dfTrain, dfTest, Dim:制限する次元数,random_state
#out:PCA変換後Train, Test, 学習後pcaモデル
def tidalPCA(dfTrain, dfTest, random_state, Dim=None):
    if Dim is None:
        pca = PCA(random_state = random_state)
    else:
        pca = PCA(n_components = Dim, random_state = random_state)
    data = pd.concat([dfTrain, dfTest])
    pca.fit(data)
    pca_train = pca.transform(dfTrain)
    pca_test = pca.transform(dfTest)
    return pca_train, pca_test, pca



#threshold(defaultは0.5)より低い分散である特徴量をdropする。
#In1  :df, trainとtest dataの連結が望ましい。(data = trainFeature.append(testFeature))
#In2  :threshold, どのくらいの大きさの分散までdropするか。(defaultは0.5)
def tidalVarianceThrs(df, threshold=0.5):
    var_thresh = VarianceThreshold(threshold=threshold)
    dfTransformed = var_thresh.fit_transform(df)
    return dfTransformed


#サンプルデータ生成
def create_dataset(n_sample=1000):
    ''' 
    Create a unevenly distributed sample data set multilabel  
    classification using make_classification function
    
    args
    nsample: int, Number of sample to be created
    
    return
    X: pandas.DataFrame, feature vector dataframe with 10 features 
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    X, y = make_classification(n_classes=5, class_sep=2,
                               weights=[0.1,0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                               n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y



def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.]) -> list:
    """
    Find the underrepresented targets.
    Underrepresented targets are those which are observed less than the median occurance.
    Targets beyond a quantile limit are filtered.
    
    不足ターゲットを見つける。
    ターゲットの頻度を分布として扱い、分位パラメータ(default:[0.05, 1.])の外に出たターゲットを返却する。
    """
    irlbl = df.sum(axis=0)
    #print(irlbl)
    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering
    irlbl = irlbl.max() / irlbl
    threshold_irlbl = irlbl.median()
    tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
    return tail_label



def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
    """
    get_tail_labelを起動し、不足しているデータを返す。
    結果をDataFrame化する。
    
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    tail_labels = get_tail_label(y, ql=ql)
    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
    
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub



def nearest_neighbour(X: pd.DataFrame, neigh) -> list:
    """
    Give index of 10 nearest neighbor of all the instance
    
    sklearnメソッド”kneighbors”の起動。
    k近傍法の実施。
    各データポイントについて、そのデータポイントに近い順にインデックス(行)を探索する。
    (よって、各行の第０列はその行自身になる。)
    
    args
    X: pandas.DataFrame, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices



def MLSMOTE(X, y, n_sample, neigh=5):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, feature vector DataFrame
    y: pandas.DataFrame, target vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    #各データポイントについて、そのデータポイントに近い順にインデックス(行)を探索する。
    indices2 = nearest_neighbour(X, neigh=5)
    
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    #augmentするデータ数分繰り返す
    for i in range(n_sample):
        #ランダムに行をピックアップ(参照行)
        reference = random.randint(0, n-1)
        #ランダムに列をピックアップ(参照行に近いインデックスをひとつ取り出す)
        neighbor = random.choice(indices2[reference, 1:])
        
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)] #参照行に近いインデックスのターゲット
        ser = nn_df.sum(axis = 0, skipna = True) #各ターゲットの和
        #各ターゲットに一つでも陽(1)が入ってれば、そのターゲット列にフラグを立てる。⇨生成データのターゲットとする。
        target[i] = np.array([1 if val > 0 else 0 for val in ser])
        ratio = random.random() #0-1のランダムな値
        
        #参照行と近いインデックスのベクトルとしての差を取る
        gap = X.loc[reference,:] - X.loc[neighbor,:]
        #参照行と近いインデックスの間の線分内のポイントをランダムに取り出す。 ⇨生成データの特徴量とする。
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_X, target



#任意の分布を持つデータをgauss分布に寄せていく。
#Input: df(trainであることが多い), testdf(定義しなければスキップ),
#       n_quantiles:計算の細かさ(累積分布の積分計算時のdxの小ささ)。値が大きくなるほど大げさにガウス分布に近づける。
#       random_state:ランダム係数
def rankGauss(dfTrain, dfTest=None, n_quantiles=100, random_state=0):
    #transformer定義
    transformer = QuantileTransformer(n_quantiles=n_quantiles, random_state=random_state, output_distribution="normal")
    
    #データ数, 列名
    vec_len = len(dfTrain.values)
    clmnNmTrain = dfTrain.columns.values[0]
    if dfTest is not None:
        vec_len_test = len(dfTest.values)
        clmnNmTest = dfTest.columns.values[0]
    
    #fitting
    raw_vec = dfTrain.values.reshape(vec_len, 1)
    transformer.fit(raw_vec)
    
    #変換
    dfTrain = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    if dfTest is not None:
        raw_vec_test = dfTest.values.reshape(vec_len_test, 1)
        dfTest = transformer.transform(raw_vec_test).reshape(1, vec_len_test)[0]
    
    if dfTest is not None:
        return pd.DataFrame(dfTrain, columns=[clmnNmTrain]), pd.DataFrame(dfTest, columns=[clmnNmTest])
    else:
        return pd.DataFrame(dfTrain, columns=[clmnNmTrain]), None