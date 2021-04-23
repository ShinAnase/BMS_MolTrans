import numpy as np

from efficientnet_210423.module.conf import setting
from efficientnet_210423.module.run.singleFoldRunning import train_loop


def Exec(NFOLDS, folds):

    for fold in range(NFOLDS):
        print('=' * 20, 'Fold', fold, '=' * 20)
        if fold in setting.trn_fold:
            train_loop(folds, fold)

    return