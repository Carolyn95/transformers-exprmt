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
  emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
  if non_trainable:
    emb_layer.weight.requires_grad = False

  return emb_layer, num_embeddings, embedding_dim


class ToyNN(nn.Module):

  def __init__(self, weights_matrix, n_classes):
    super(ToyNN, self).__init__()
    self.embedding, num_embeddings, embedding_dim = create_emb_layer(
        weights_matrix, True)
    self.n_classes = n_classes
    self.hid = nn.Linear(embedding_dim, n_classes)
    self.softmax_act = nn.Softmax(dim=1)

  def forward(self, text):
    pdb.set_trace()
    output = self.embedding(text)

    logits = self.softmax_act(self.hid(output))
    # loss_fct = nn.CrossEntropyLoss()
    # loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))
    # return {"loss": loss, "logits": logits}
    return logits


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
    for sent_text, label in zip(sents, labels):
      # inputs = self.tokenizer(sent_text, add_special_tokens=True)
      # inputs['labels'] = torch.tensor(labels[idx],
      #                                 dtype=torch.long).unsqueeze(0)
      temp = {"text": sent_text, "label": label}
      self.datasets.append(temp)


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
model = ToyNN(weights_matrix, n_classes)

datasets = {}
datasets['train'] = train_datasets
datasets['test'] = dev_datasets

is_grad = 'grad_'


def compute_metrics(pred):
  labels = pred.label_ids
  preds = [x.argmax(-1) for x in pred.predictions]
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


"""
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
"""

import time


def train(dataloader):
  model.train()
  total_acc, total_count = 0, 0
  log_interval = 500
  start_time = time.time()

  for idx, (text, label) in enumerate(dataloader):
    optimizer.zero_grad()
    predited_label = model(text)
    loss = criterion(predited_label, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    total_acc += (predited_label.argmax(1) == label).sum().item()
    total_count += label.size(0)
    if idx % log_interval == 0 and idx > 0:
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches '
            '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                        total_acc / total_count))
      total_acc, total_count = 0, 0
      start_time = time.time()


def evaluate(dataloader):
  model.eval()
  total_acc, total_count = 0, 0

  with torch.no_grad():
    for idx, (label, text, offsets) in enumerate(dataloader):
      predited_label = model(text, offsets)
      loss = criterion(predited_label, label)
      total_acc += (predited_label.argmax(1) == label).sum().item()
      total_count += label.size(0)
  return total_acc / total_count


#  TRAINING & EVALUATING
# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

for epoch in range(1, EPOCHS + 1):
  epoch_start_time = time.time()
  train(datasets['train'])
  accu_val = evaluate(datasets['test'])
  if total_accu is not None and total_accu > accu_val:
    scheduler.step()
  else:
    total_accu = accu_val
  print('-' * 59)
  print('| end of epoch {:3d} | time: {:5.2f}s | '
        'valid accuracy {:8.3f} '.format(epoch,
                                         time.time() - epoch_start_time,
                                         accu_val))
  print('-' * 59)
