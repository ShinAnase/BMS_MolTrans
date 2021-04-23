import numpy as np
from sklearn import preprocessing
import optuna
import time
import torch

#自作app
from module import loading, CV_folds
from module import runKFolds
from efficientnet_210423.module.conf import setting

import warnings
warnings.filterwarnings('ignore')


def main(trial):
    #Hyper parameter
    param = {'hidden_size1': trial.suggest_categorical('hidden_size1', [128, 256, 512]),
             'dropOutRate1': trial.suggest_uniform('dropOutRate1', 0.01, 0.5),
             }

    #debug mode
    if setting.debug:
        setting.epochs = 1
        train = loading.train.sample(n=1000, random_state=42).reset_index(drop=True)

    # Preprocessing Data
    #train, test, targetOH = preprocessing.Exec(param, loading.trainFeature,
    #                                         loading.testFeature, loading.trainTarget)

    # CV folds
    folds = CV_folds.Exec(loading.train)

    ### RUN ###
    for seed in setting.seed:
        print('~' * 20, 'SEED', seed, '~' * 20)
        runKFolds.Exec(setting.n_fold, folds)





start = time.time()#時間計測用タイマー開始
main(optuna.trial.FixedTrial({'hidden_size1': 256,
                                      'dropOutRate1': 0.3}))
#tuning
#study = optuna.create_study()
#study.optimize(Exec, n_trials=7)
#print(f"best param: {study.best_params}")
#print(f"best score: {study.best_value}")

elapsed_time = (time.time() - start)/60/60
print(f"Time：{elapsed_time}[h]")
