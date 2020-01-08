# -*- coding: utf-8 -*-
__author__ = 'yshao'

import torch
import numpy as np
import pandas as pd
import matchzoo as mz
import os
print('matchzoo version', mz.__version__)

DATA_DIR = '/data/disk2/private/guozhipeng/syq/coliee/Case_Law/format/matchzoo'

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=1))
ranking_task.metrics = [
    mz.metrics.Precision(k=5),
    mz.metrics.Recall(k=5),
    mz.metrics.F1(k=5)
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

train_pack_raw = mz.pack(pd.read_csv(os.path.join(DATA_DIR, 'train_bm25.csv'), index_col=False, encoding='utf8'), 'ranking')
dev_pack_raw = mz.pack(pd.read_csv(os.path.join(DATA_DIR, 'dev_bm25.csv'), index_col=False, encoding='utf8'), 'ranking')
test_pack_raw = mz.pack(pd.read_csv(os.path.join(DATA_DIR, 'test_bm25.csv'), index_col=False, encoding='utf8'), 'ranking')

print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

preprocessor = mz.models.MatchPyramid.get_default_preprocessor(
    filter_mode='df',
    filter_low_freq=2,
)

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

print(preprocessor.context)

glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=1,
    num_neg=5
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed
)

validset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed
)

padding_callback = mz.models.MatchPyramid.get_default_padding_callback(
    fixed_length_left=256,
    fixed_length_right=256,
    pad_word_value=0,
    pad_word_mode='pre'
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
    dataset=validset,
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

model = mz.models.MatchPyramid()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix

model.params['kernel_count'] = [16, 32]
model.params['kernel_size'] = [[3, 3], [3, 3]]
model.params['dpool_size'] = [3, 10]
model.params['dropout_rate'] = 0.1

model.build()

print(model, sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters())

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

