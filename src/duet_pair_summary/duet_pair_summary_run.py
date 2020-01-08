# -*- coding: utf-8 -*-
__author__ = 'yshao'


import torch
import numpy as np
import pandas as pd
import matchzoo as mz
import os
print('matchzoo version', mz.__version__)

DATA_DIR = '/data/disk2/private/guozhipeng/syq/coliee/Case_Law/format/matchzoo'

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.Precision(k=5),
    mz.metrics.Recall(k=5),
    mz.metrics.F1(k=5)
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

train_pack_raw = mz.pack(pd.read_csv(os.path.join(DATA_DIR, 'train_256_bm25.csv'), index_col=False, encoding='utf8'), 'ranking')
dev_pack_raw = mz.pack(pd.read_csv(os.path.join(DATA_DIR, 'dev_256_bm25.csv'), index_col=False, encoding='utf8'), 'ranking')
test_pack_raw = mz.pack(pd.read_csv(os.path.join(DATA_DIR, 'test_256_bm25.csv'), index_col=False, encoding='utf8'), 'ranking')

print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

preprocessor = mz.models.DUET.get_default_preprocessor(
    filter_mode='df',
    filter_low_freq=2,
    truncated_mode='post',
    truncated_length_left=256,
    truncated_length_right=256,
    ngram_size=3
)

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

print(preprocessor.context)

triletter_callback = mz.dataloader.callbacks.Ngram(preprocessor, mode='sum')
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=1,
    num_neg=5,
    callbacks=[triletter_callback]
)
devset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed,
    callbacks=[triletter_callback]
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    callbacks=[triletter_callback]
)

padding_callback = mz.models.DUET.get_default_padding_callback(
    fixed_length_left=256,
    fixed_length_right=256,
    pad_word_value=0,
    pad_word_mode='pre',
    with_ngram=True,
)

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    batch_size=16,
    stage='train',
    sort=False,
    shuffle=True,
    callback=padding_callback,
    num_workers=4
)

validloader = mz.dataloader.DataLoader(
    dataset=devset,
    batch_size=8,
    stage='dev',
    sort=False,
    callback=padding_callback,
    num_workers=2
)

testloader = mz.dataloader.DataLoader(
    dataset=testset,
    batch_size=8,
    stage='dev',
    sort=False,
    callback=padding_callback,
    num_workers=2
)

model = mz.models.DUET()

model.params['task'] = ranking_task
model.params['left_length'] = 256
model.params['right_length'] = 256
model.params['lm_filters'] = 100
model.params['mlp_num_layers'] = 2
model.params['mlp_num_units'] = 100
model.params['mlp_num_fan_out'] = 100
model.params['mlp_activation_func'] = 'tanh'

model.params['vocab_size'] = preprocessor.context['ngram_vocab_size']
model.params['dm_conv_activation_func'] = 'relu'
model.params['dm_filters'] = 100
model.params['dm_kernel_size'] = 3
model.params['dm_right_pool_size'] = 4
model.params['dropout_rate'] = 0.2


model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adadelta(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    testloader=testloader,
    validate_interval=None,
    epochs=10
)

trainer.run()
