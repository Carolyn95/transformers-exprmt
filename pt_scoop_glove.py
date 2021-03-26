# torchtext==0.9.1
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pickle as pkl
import torchtext.data
from torchtext.legacy.data import Field
from torchtext.legacy.data import Dataset, Example
from torchtext.legacy.data import BucketIterator
from torchtext.vocab import FastText
from torchtext.vocab import CharNGram
from sklearn.metrics import accuracy_score
from torchtext.vocab import GloVe

import pandas as pd
import numpy as np
# embedding = FastText('simple')
embedding = GloVe(name='6B', dim=50)

data_dir = './data/atfm-data/'
train_ = np.load(data_dir + 'train_sents.npy', allow_pickle=True)
train_labels = np.load(data_dir + 'labels_train.npy', allow_pickle=True)
eval_ = np.load(data_dir + 'eval_sents.npy', allow_pickle=True)
eval_labels = np.load(data_dir + 'labels_val.npy', allow_pickle=True)

texts = np.concatenate((train_, eval_))
labels = np.concatenate((train_labels, eval_labels))

df = pd.DataFrame({'text': texts, 'label': labels})

text_field = Field(sequential=True,
                   tokenize='basic_english',
                   fix_length=5,
                   lower=True)

label_field = Field(sequential=False, use_vocab=False, is_target=True)

preprocessed_text = df['text'].apply(lambda x: text_field.preprocess(x))
text_field.build_vocab(preprocessed_text, vectors='fasttext.simple.300d')
vocab = text_field.vocab

ltoi = {l: i for i, l in enumerate(df['label'].unique())}
df['label'] = df['label'].apply(lambda y: ltoi[y])


class DataFrameDataset(torchtext.legacy.data.Dataset):

  def __init__(self, df: pd.DataFrame, fields: list):
    super(DataFrameDataset, self).__init__(
        [Example.fromlist(list(r), fields) for i, r in df.iterrows()], fields)


train_dataset, test_dataset = DataFrameDataset(df=df,
                                               fields=(('text', text_field),
                                                       ('label',
                                                        label_field))).split()
with open(data_dir + 'n_labels.pkl', 'rb') as f:
  n_classes = pkl.load(f)
train_iter, test_iter = BucketIterator.splits(datasets=(train_dataset,
                                                        test_dataset),
                                              batch_sizes=(32, n_classes),
                                              sort=False)


class ModelParam(object):

  def __init__(self, param_dict: dict = dict()):
    self.input_size = param_dict.get('input_size', 0)
    self.vocab_size = param_dict.get('vocab_size')
    self.embedding_dim = param_dict.get('embedding_dim', 300)
    self.target_dim = param_dict.get('target_dim', n_classes)


class MyModel(nn.Module):

  def __init__(self, model_param: ModelParam):
    super().__init__()
    self.embedding = nn.Embedding(model_param.vocab_size,
                                  model_param.embedding_dim)
    self.lin = nn.Linear(model_param.input_size * model_param.embedding_dim,
                         model_param.target_dim)

  def forward(self, x):
    features = self.embedding(x).view(x.size()[0], -1)
    features = F.relu(features)
    features = self.lin(features)
    return features


class MyModelWithPretrainedEmbedding(nn.Module):

  def __init__(self, model_param: ModelParam, embedding):
    super().__init__()
    self.embedding = embedding
    self.lin = nn.Linear(model_param.input_size * model_param.embedding_dim,
                         model_param.target_dim)

  def forward(self, x):
    features = self.embedding[x].reshape(x.size()[0], -1)
    features = F.relu(features)
    features = self.lin(features)
    return features


model_param = ModelParam(
    param_dict=dict(vocab_size=len(text_field.vocab), input_size=5))
model = MyModel(model_param)
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)
epochs = 10

for epoch in range(epochs):
  epoch_losses = list()
  for batch in train_iter:
    optimizer.zero_grad()

    prediction = model(batch.text.T)
    # pdb.set_trace()
    loss = loss_function(prediction, batch.label)

    loss.backward()
    optimizer.step()

    epoch_losses.append(loss.item())
  print('train loss on epoch {} : {:.3f}'.format(epoch, np.mean(epoch_losses)))

  test_losses = list()
  for batch in test_iter:
    with torch.no_grad():
      optimizer.zero_grad()
      prediction = model(batch.text.T)
      loss = loss_function(prediction, batch.label)

      test_losses.append(loss.item())

      acc = accuracy_score(batch.label, prediction.argmax(-1))

  print('test loss on epoch {}: {:.3f}\ntest acc: {:.3f}'.format(
      epoch, np.mean(test_losses), acc))
