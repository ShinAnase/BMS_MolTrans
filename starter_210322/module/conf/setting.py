import torch

INPUT = "/home/tidal/ML_Data/BMS_MolTrans/bms-molecular-translation"
OUTPUT = "/home/tidal/ML_Data/BMS_MolTrans/output"

SUBMIT = OUTPUT + "/submittion/"
SAVEMODEL = OUTPUT + "/model/Pytorch/"
SAVEOOF = OUTPUT + "/OOF/Pytorch/"
SAVEPLOT = OUTPUT + "/plot_history/"
SAVEIMG = OUTPUT + "/plot_img/"

debug=False
max_len=275
print_freq=1000
num_workers=4
model_name='resnet34'
size=224
scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
epochs=1 # not to exceed 9h
#factor=0.2 # ReduceLROnPlateau
#patience=4 # ReduceLROnPlateau
#eps=1e-6 # ReduceLROnPlateau
T_max=4 # CosineAnnealingLR
#T_0=4 # CosineAnnealingWarmRestarts
encoder_lr=1e-4
decoder_lr=4e-4
min_lr=1e-6
batch_size=64
weight_decay=1e-6
gradient_accumulation_steps=1
max_grad_norm=5
attention_dim=256
embed_dim=256
decoder_dim=512
dropout=0.5
seed=[42]
n_fold=5
trn_fold=[0] # [0, 1, 2, 3, 4]
train=True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#logger
def init_logger(log_file=OUTPUT+'/Log/train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()

