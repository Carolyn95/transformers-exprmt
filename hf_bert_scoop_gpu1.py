"""
20210310                       
sequence classification with scoop dataset (atfm, ssoe, sats)
data input pair (send, int_label)
cls head (sequence classification provided by huggingface)
eval use sklearn acc
take last hidded states from AutoModelForSequenceClassification, apply mean pooling

first try implement a customed trainer class, but too complicated 
go for modifying packages
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pdb
import datasets
import random
import pandas as pd
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(2021)


class ScoopData(Dataset):

  def __init__(self, data_dir, tokenizer, train_or_test='train', max_len=32):
    self.data_file_path = data_dir
    self.train_or_test = train_or_test
    self.tokenizer = tokenizer
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
    for idx, sent_text in enumerate(sents):
      # `return_token_type_ids=False` for distilbert
      # `return_token_type_ids=True` for bert
      temp = {}
      inputs = self.tokenizer.encode_plus(sent_text,
                                          None,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          pad_to_max_length=True,
                                          return_token_type_ids=False)

      inputs['labels'] = torch.tensor(labels[idx],
                                      dtype=torch.long).unsqueeze(0)
      self.datasets.append(inputs)


def getDataset(data_dir, tokenizer, train_or_test='train', is_shuffle=True):
  dataset = ScoopData(data_dir=data_dir,
                      tokenizer=tokenizer,
                      train_or_test=train_or_test)
  dataset_loader = DataLoader(dataset,
                              batch_size=16,
                              shuffle=is_shuffle,
                              drop_last=True,
                              num_workers=2).dataset
  return dataset_loader


# "distilbert-base-uncased" | "distilbert-base-cased" | "bert-base-uncased" | "bert-base-cased"
# "funnel-transformer/small-base" | "funnel-transformer/intermediate" | "funnel-transformer/xlarge-base" | "funnel-transformer/medium-base" | "funnel-transformer/large-base"

model_checkpoint = "distilbert-base-uncased"
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_dir = './data/atfm-data/'
train_datasets = getDataset(data_dir=data_dir,
                            tokenizer=tokenizer,
                            train_or_test='train',
                            is_shuffle=True)
dev_datasets = getDataset(data_dir=data_dir,
                          tokenizer=tokenizer,
                          train_or_test='test',
                          is_shuffle=True)
label_list = train_datasets.all_labels
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list), output_hidden_states=True)
datasets = {}
datasets['train'] = train_datasets
datasets['test'] = dev_datasets

is_grad = 'grad_'
# for name, param in model.named_parameters():
#   if 'classifier' not in name:  # classifier layer
#     param.requires_grad = False

from sklearn.metrics import accuracy_score


def compute_metrics(pred):
  labels = pred.label_ids
  # pdb.set_trace()
  # preds = pred.predictions.argmax(-1)
  preds = pred.predictions[0].argmax(-1)
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


batch_size = 32
exprmt_ds = data_dir.split('/')[-2].split('-')[0]
traning_samples = len(train_datasets)
output_dir = 'hf_' + is_grad + f'{model_name}' + f'_{exprmt_ds}' + f'_{traning_samples}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    model_parallel=True
)  # model_parallel should use in conjunction with model.to('cuda')

# hidden_size = model.config.hidden_size
# num_labels = len(label_list)

# from torch import nn
# from torch.nn import CrossEntropyLoss
# class MeanPooledTrainer(Trainer):

#   def create_cls(self, hidden_size, num_labels):
#     self.hidden_size = hidden_size
#     self.num_labels = num_labels
#     self.cls = nn.Linear(hidden_size, num_labels)

#   def compute_loss(self, model, inputs):
#     labels = inputs.pop("labels")
#     outputs = model(**inputs)
#     pooled_output = torch.mean(ouputs.hidden_states[-1], 1)
#     self.cls = nn.Linear(hidden_size, num_labels)
#     logits = self.cls(pooled_output)
#     loss_fct = CrossEntropyLoss()
#     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#     return loss

trainer = Trainer(model.to(device),
                  args,
                  train_dataset=datasets['train'],
                  eval_dataset=datasets['train'],
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)
# trainer = MeanPooledTrainer(model.to(device),
#                   args,
#                   train_dataset=datasets['train'],
#                   eval_dataset=datasets['train'],
#                   tokenizer=tokenizer,
#                   compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(args.output_dir)

predictions, labels, _ = trainer.predict(datasets['test'])
predictions = np.argmax(predictions[0], axis=1)
results = accuracy_score(labels, predictions)
print(results)
