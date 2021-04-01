import pickle
import numpy as np
import pandas as pd

from starter_210322.module.conf import setting


# Loading
def get_train_file_path(image_id):
    return setting.INPUT + "/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id
    )


train_ = pd.read_pickle(setting.OUTPUT + '/preprocessingTrain/train2_pycharm.pkl')
train_['file_path'] = train_['image_id'].apply(get_train_file_path)
train = train_.copy()
del train_
