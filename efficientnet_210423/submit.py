import gc

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from efficientnet_210423.module import loading
from efficientnet_210423.module.conf import setting
from efficientnet_210423.module.dataset import TestDataset
from efficientnet_210423.module.run.arch.CNN import Encoder
from efficientnet_210423.module.run.arch.LSTM import DecoderWithAttention
from efficientnet_210423.module.transform import get_transforms


def inference(test_loader, encoder, decoder, tokenizer, device):
    encoder.eval()
    decoder.eval()
    text_preds = []
    tk0 = tqdm(test_loader, total=len(test_loader))
    for images in tk0:
        images = images.to(device)
        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, setting.max_len, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)
    text_preds = np.concatenate(text_preds)
    return text_preds


# -------------------------submit実行---------------------------------------------
with open(setting.OUTPUT + '/Log/train.log') as f:
    s = f.read()
print(s)

tokenizer = torch.load(setting.OUTPUT + '/tokenizer/tokenizer2_pycharm.pth')
print(f"tokenizer.stoi: {tokenizer.stoi}")

states = torch.load(f'{setting.OUTPUT}/model/{setting.model_name}_fold0_best.pth',
                    map_location=torch.device('cpu'))

encoder = Encoder(setting.model_name, pretrained=False)
encoder.load_state_dict(states['encoder'])
encoder.to(setting.device)

decoder = DecoderWithAttention(attention_dim=setting.attention_dim,
                               embed_dim=setting.embed_dim,
                               decoder_dim=setting.decoder_dim,
                               vocab_size=len(tokenizer),
                               dropout=setting.dropout,
                               device=setting.device)
decoder.load_state_dict(states['decoder'])
decoder.to(setting.device)

del states;
gc.collect()

test_dataset = TestDataset(loading.test, transform=get_transforms(data='valid'))
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=setting.num_workers)
predictions = inference(test_loader, encoder, decoder, tokenizer, setting.device)

del test_loader, encoder, decoder, tokenizer;
gc.collect()

# submission
sub = loading.test.copy()
sub['InChI'] = [f"InChI=1S/{text}" for text in predictions]
sub[['image_id', 'InChI']].to_csv(setting.OUTPUT + '/submittion/submission.csv', index=False)
sub[['image_id', 'InChI']].head()
