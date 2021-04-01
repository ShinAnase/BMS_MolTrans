import numpy as np

from starter_210322.module.conf import setting
from starter_210322.module.run.singleFoldRunning import train_loop


def Exec(NFOLDS, folds):

    for fold in range(NFOLDS):
        print('=' * 20, 'Fold', fold, '=' * 20)
        if fold in setting.trn_fold:
            train_loop(folds, fold)

    return