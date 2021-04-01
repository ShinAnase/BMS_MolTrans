from sklearn.model_selection import StratifiedKFold
from starter_210322.module.conf import setting


def Exec(train):
    folds = train.copy()
    Fold = StratifiedKFold(n_splits=setting.n_fold, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['InChI_length'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)

    return folds