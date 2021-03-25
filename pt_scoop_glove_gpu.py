# 20210325 ❤ Carolyn CHEN
# simply try on glove, glove => word embedding => mean pooling over words to get sentence representation => cls
# ? glove and word2vec
import bcolz
import pickle
import pdb
import numpy as np
import itertools
from torch import nn
import torch
import random
import time
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(2021)

glove_path = 'glove_emb'

# >>>>> GET GLOVE >>>>>
# words = []
# idx = 0
# word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

# with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
#   for l in f:
#     line = l.decode().split()
#     word = line[0]
#     words.append(word)
#     word2idx[word] = idx
#     idx += 1
#     vect = np.array(line[1:]).astype(np.float)
#     vectors.append(vect)

# vectors = bcolz.carray(vectors[1:].reshape((400000, 50)),
#                        rootdir=f'{glove_path}/6B.50.dat',
#                        mode='w')
# vectors.flush()
# pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
# pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}
# TEST ----
# print(glove['pilot'])
# pdb.set_trace()
data_dir = './data/sats-data/'

train_ = np.load(data_dir + 'train_sents.npy', allow_pickle=True)
eval_ = np.load(data_dir + 'eval_sents.npy', allow_pickle=True)
total = np.concatenate((train_, eval_))
target_vocab = list(set(itertools.chain(*[sent.split() for sent in total])))
matrix_len = len(target_vocab)
embed_sz = 50
weights_matrix = np.zeros((matrix_len, embed_sz))
words_found = 0

for i, word in enumerate(target_vocab):
  try:
    weights_matrix[i] = glove[word]
    words_found += 1
  except KeyError:
    weights_matrix[i] = np.random.normal(scale=0.6, size=(embed_sz,))


def create_emb_layer(weights_matrix, non_trainable=False):
  num_embeddings, embedding_dim = weights_matrix.shape
  emb_layer = nn.Embedding(num_embeddings, embedding_dim)
  # pdb.set_trace()
  emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
  if non_trainable:
    emb_layer.weight.requires_grad = False

  return emb_layer, num_embeddings, embedding_dim


class ToyNN(nn.Module):

  def __init__(self, weights_matrix, hidden_size, num_layers):
    super(ToyNN, self).__init__()
    self.embedding, num_embeddings, embedding_dim = create_emb_layer(
        weights_matrix, True)
    # self.n_classes = n_classes
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
    # self.hid1 = nn.Linear(768, 256)
    # self.hid2 = nn.Linear(256, 256)
    # self.out = nn.Linear(256, self.n_classes)
    # self.relu_act = nn.ReLU()
    # self.tanh_act = nn.Tanh()
    # self.softmax_act = nn.Softmax(dim=1)
    # self.drop = nn.Dropout(p=0.3)

  def forward(self, inp, hidden):
    return self.gru(self.embedding(inp), hidden)

  def init_hidden(self, batch_size):
    return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))


model = ToyNN(weights_matrix, 256, 6)


class ScoopData(Dataset):

  def __init__(self, data_dir, train_or_test='train', max_len=32):
    self.data_file_path = data_dir
    self.train_or_test = train_or_test
    self.max_len = max_len
    self._build()

  def __getitem__(self, index):
    return self.datasets[index]

  def __len__(self):
    return len(self.datasets)

  def _build(self):
    self._buil_examples_from_files(self.data_file_path)

  def _buil_examples_from_files(self, data_file_path):

    if self.train_or_test == 'train':
      sents = np.load(data_dir + 'train_sents.npy', allow_pickle=True)
      labels = np.load(data_dir + 'int_labels_train.npy', allow_pickle=True)
    else:
      sents = np.load(data_dir + 'eval_sents.npy', allow_pickle=True)
      labels = np.load(data_dir + 'int_labels_val.npy', allow_pickle=True)

    self.all_labels = np.unique(labels, axis=0)
    self.datasets = []
    for sent_text in sents:
      # inputs = self.tokenizer(sent_text, add_special_tokens=True)
      # pdb.set_trace()
      # inputs['labels'] = torch.tensor(labels[idx],
      #                                 dtype=torch.long).unsqueeze(0)
      self.datasets.append(sent_text)


def getDataset(data_dir, train_or_test='train', is_shuffle=True):
  dataset = ScoopData(data_dir=data_dir, train_or_test=train_or_test)
  dataset_loader = DataLoader(dataset,
                              batch_size=16,
                              shuffle=is_shuffle,
                              drop_last=True,
                              num_workers=2).dataset
  return dataset_loader


# training loop

train_datasets = getDataset(data_dir=data_dir,
                            train_or_test='train',
                            is_shuffle=True)
dev_datasets = getDataset(data_dir=data_dir,
                          train_or_test='test',
                          is_shuffle=True)
label_list = train_datasets.all_labels
n_classes = len(label_list)

datasets = {}
datasets['train'] = train_datasets
datasets['test'] = dev_datasets

is_grad = 'grad_'


def compute_metrics(pred):
  labels = pred.label_ids
  preds = [x.argmax(-1) for x in pred.predictions]
  # pdb.set_trace()
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


batch_size = 32
exprmt_ds = data_dir.split('/')[-2].split('-')[0]
traning_samples = len(train_datasets)
model_name = 'glove'
output_dir = 'hf_' + is_grad + f'{model_name}' + f'_{exprmt_ds}' + f'_{traning_samples}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = TrainingArguments(output_dir=output_dir,
                         evaluation_strategy='epoch',
                         learning_rate=2e-5,
                         per_device_train_batch_size=batch_size,
                         per_device_eval_batch_size=batch_size,
                         num_train_epochs=10,
                         weight_decay=0.01,
                         model_parallel=True)

trainer = Trainer(model.to(device),
                  args,
                  train_dataset=datasets['train'],
                  eval_dataset=datasets['train'],
                  compute_metrics=compute_metrics)

trainer.train()
trainer.save_model(args.output_dir)

predictions, labels, _ = trainer.predict(datasets['test'])
# pdb.set_trace()
predictions = np.argmax(predictions, axis=1)
results = accuracy_score(labels, predictions)
print(results)
