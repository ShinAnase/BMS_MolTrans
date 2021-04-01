import numpy as np
import pandas as pd

#targetのカラム
tarClmn = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def Collecting(trainFeature, testFeature, trainTarget):
    # targetを結合
    trainFeature = pd.concat([trainFeature, trainTarget], axis=1)
    # test側のtargetは0で初期化しておく
    testTarTmp = pd.DataFrame(np.zeros((testFeature.shape[0], trainTarget.shape[1]), dtype='uint8'), columns=tarClmn)
    testFeature = pd.concat([testFeature, testTarTmp], axis=1)

    return trainFeature, testFeature


#Preprocessing Summary
def Exec(param, trainFeature, testFeature, trainTarget):
    # targetのデータフレーム作成（列名も付与）(one-hot化しておく)
    targetOH = pd.get_dummies(trainTarget)
    targetOH.columns = tarClmn

    # train,testにターゲット値を連結。
    train, test = Collecting(trainFeature, testFeature, targetOH)

    return train, test, targetOH