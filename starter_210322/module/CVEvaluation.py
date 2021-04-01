from sklearn.metrics import log_loss, accuracy_score
import numpy as np
from starter_210322.module.conf import setting


def Exec(confFitting, oof, target, targetOH):
    score = {}

    # cross entropy
    y_true_OH = targetOH[confFitting["target_cols"]].values
    y_pred_proba = oof

    score_logloss = 0
    for i in range(confFitting["num_targets"]):
        score_ = log_loss(y_true_OH[:, i], y_pred_proba[:, i])  # 問題の評価指標によって変わる。
        score_logloss += score_ / targetOH.shape[1]

    print("CV cross entropy: ", score_logloss)
    score["Logloss"] = score_logloss

    # accuracy
    score_accuracy = 0
    y_true = target.values
    y_pred = np.zeros((target.shape[0],))
    for i in range(target.shape[0]):
        # pred_proba->predに変形
        y_pred[i] = np.argmax(oof[i])

    score_accuracy = accuracy_score(y_true, y_pred)

    print("CV accuracy: ", score_accuracy)
    score["Accuracy"] = score_accuracy

    # OOF save
    np.save(setting.SAVEOOF + 'oof', y_pred_proba)

    return score