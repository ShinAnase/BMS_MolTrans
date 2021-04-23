import time

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

#自作app
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from efficientnet_210423.module import tokenizer
from efficientnet_210423.module.conf.setting import LOGGER, OUTPUT
from efficientnet_210423.module.dataset import TrainDataset, TestDataset, bms_collate
from efficientnet_210423.module.run.arch.CNN import Encoder
from efficientnet_210423.module.run.arch.LSTM import DecoderWithAttention
from efficientnet_210423.module.conf import setting


# Seed固定
from efficientnet_210423.module.run.runUtl.runFn import train_fn, valid_fn, get_score
from efficientnet_210423.module.transform import get_transforms


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ====================================================
# Train loop
# ====================================================
def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values

    train_dataset = TrainDataset(train_folds, tokenizer.tokenizer_ins, transform=get_transforms(data='train'))
    valid_dataset = TestDataset(valid_folds, transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.batch_size,
                              shuffle=True,
                              num_workers=setting.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=bms_collate)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=setting.batch_size,
                              shuffle=False,
                              num_workers=setting.num_workers,
                              pin_memory=True,
                              drop_last=False)

    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if setting.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=setting.factor, patience=setting.patience, verbose=True,
                                          eps=setting.eps)
        elif setting.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=setting.T_max, eta_min=setting.min_lr, last_epoch=-1)
        elif setting.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=setting.T_0, T_mult=1, eta_min=setting.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    encoder = Encoder(setting.model_name, pretrained=True)
    #print(f"setting.device:{setting.device}")
    encoder.to(setting.device)
    encoder_optimizer = Adam(encoder.parameters(), lr=setting.encoder_lr, weight_decay=setting.weight_decay, amsgrad=False)
    encoder_scheduler = get_scheduler(encoder_optimizer)

    decoder = DecoderWithAttention(attention_dim=setting.attention_dim,
                                   embed_dim=setting.embed_dim,
                                   decoder_dim=setting.decoder_dim,
                                   vocab_size=len(tokenizer.tokenizer_ins),
                                   dropout=setting.dropout,
                                   device=setting.device)
    decoder.to(setting.device)
    decoder_optimizer = Adam(decoder.parameters(), lr=setting.decoder_lr, weight_decay=setting.weight_decay, amsgrad=False)
    decoder_scheduler = get_scheduler(decoder_optimizer)

    # ====================================================
    # loop
    # ====================================================

    #print(f"tokenizer.stoi: {tokenizer.tokenizer_ins.stoi}")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.tokenizer_ins.stoi["<pad>"])

    best_score = np.inf
    best_loss = np.inf

    for epoch in range(setting.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, encoder, decoder, criterion,
                            encoder_optimizer, decoder_optimizer, epoch,
                            encoder_scheduler, decoder_scheduler, setting.device)

        # eval
        text_preds = valid_fn(valid_loader, encoder, decoder, tokenizer.tokenizer_ins, criterion, setting.device)
        text_preds = [f"InChI=1S/{text}" for text in text_preds]
        LOGGER.info(f"labels: {valid_labels[:5]}")
        LOGGER.info(f"preds: {text_preds[:5]}")

        # scoring
        score = get_score(valid_labels, text_preds)

        if isinstance(encoder_scheduler, ReduceLROnPlateau):
            encoder_scheduler.step(score)
        elif isinstance(encoder_scheduler, CosineAnnealingLR):
            encoder_scheduler.step()
        elif isinstance(encoder_scheduler, CosineAnnealingWarmRestarts):
            encoder_scheduler.step()

        if isinstance(decoder_scheduler, ReduceLROnPlateau):
            decoder_scheduler.step(score)
        elif isinstance(decoder_scheduler, CosineAnnealingLR):
            decoder_scheduler.step()
        elif isinstance(decoder_scheduler, CosineAnnealingWarmRestarts):
            decoder_scheduler.step()

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')

        if score < best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'encoder': encoder.state_dict(),
                        'encoder_optimizer': encoder_optimizer.state_dict(),
                        'encoder_scheduler': encoder_scheduler.state_dict(),
                        'decoder': decoder.state_dict(),
                        'decoder_optimizer': decoder_optimizer.state_dict(),
                        'decoder_scheduler': decoder_scheduler.state_dict(),
                        'text_preds': text_preds,
                        },
                       OUTPUT + f'/model/{setting.model_name}_fold{fold}_best.pth')