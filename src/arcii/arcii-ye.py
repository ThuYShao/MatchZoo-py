import torch
import nltk
nltk.data.path.append('/work/yeziyi/nltk_data')
import numpy as np
import pandas as pd
import matchzoo as mz
print('matchzoo version', mz.__version__)

ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=1),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=10),
    mz.metrics.Precision(k=1, threshold = 0.0),
    mz.metrics.Precision(k=3),
    mz.metrics.Precision(k=5),
    mz.metrics.MeanAveragePrecision(),
    mz.metrics.MeanReciprocalRank(threshold=0),
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)

train_pack_raw = mz.pack(pd.read_csv('../../data2/train1.csv', index_col=0, encoding='utf8'), 'ranking')
dev_pack_raw = mz.pack(pd.read_csv('../../data2/dev1.csv', index_col=0, encoding='utf8'), 'ranking')
test_pack_raw = mz.pack(pd.read_csv('../../data2/test.csv', index_col=0, encoding='utf8'), 'ranking')

print('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

preprocessor = mz.models.ArcII.get_default_preprocessor()

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)

glove_embedding = mz.embedding.load_from_file('/work/chenjia/Hybrid_context_modeling/HCSM/data/vectors.txt', mode='glove')


term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

# In[6]:

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
# In[7]:


padding_callback = mz.models.ArcII.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    batch_size=4,
    stage='train',
    resample=True,
    sort=False,
    callback=padding_callback,
    num_workers = 1,
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    batch_size=4,
    stage='dev',
    callback=padding_callback,
    num_workers=1,
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    batch_size=4,
    stage='dev',
    callback=padding_callback,
    num_workers=1,
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

)


# In[8]:


model = mz.models.ArcII()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['left_length'] = 10
model.params['right_length'] = 100
model.params['kernel_1d_count'] = 32
model.params['kernel_1d_size'] = 3
model.params['kernel_2d_count'] = [64, 64]
model.params['kernel_2d_size'] = [(3, 3), (3, 3)]
model.params['pool_2d_size'] = [(3, 3), (3, 3)]

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# In[9]:

optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    testloader=testloader,
    validate_interval=None,
    epochs=15,
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # checkpoint = 'save/model.pt'
)


# In[10]:


trainer.run()
