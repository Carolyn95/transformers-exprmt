"""
  20210326
  banking 77
  huggingface pipeline validation
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
import csv
import json
from pathlib import Path
from collections import Counter


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(2021)


class Banking77(datasets.GeneratorBasedBuilder):

  def _info(self):
    return datasets.info.DatasetInfo()

  def _split_generators(self, dl_manager):
    data = dl_manager.download_and_extract({
        split: Path(self.config.data_dir).joinpath(f'{split}.csv')
        for split in ['train', 'test']
    })
    with open(data['train']) as fp:
      all_train = [x for x in csv.DictReader(fp)]
    with open(data['test']) as fp:
      test = [x for x in csv.DictReader(fp)]
    val_split = []
    labels = [x['category'] for x in all_train]
    for label, count in Counter(labels).items():
      n = count // 5
      val_split.extend([i for i, c in enumerate(labels) if c == label][:n])
    train_split = [i for i in range(len(all_train)) if i not in val_split]
    return [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={'samples': [all_train[i] for i in train_split]}),
        datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={'samples': [all_train[i] for i in val_split]}),
        datasets.SplitGenerator(name=datasets.Split.TEST,
                                gen_kwargs={'samples': test})
    ]

  def _generate_examples(self, samples):
    for i, eg in enumerate(samples):
      yield i, {'id': str(i), 'text': eg['text'], 'label': eg['category']}

  @staticmethod
  def load(data_dir=None, **kwargs):
    src_file = Path(__file__)
    if not data_dir:
      data_dir = src_file.with_suffix('')
    with open(Path(data_dir).joinpath('categories.json')) as fp:
      features = datasets.Features({
          'id': datasets.Value('string'),
          'text': datasets.Value('string'),
          'label': datasets.features.ClassLabel(names=json.load(fp))
      })
    return datasets.load_dataset(str(src_file.absolute()),
                                 data_dir=data_dir,
                                 features=features,
                                 **kwargs)


# "distilbert-base-uncased" | "distilbert-base-cased"
# "bert-base-uncased" | "bert-base-cased"

model_checkpoint = "bert-base-uncased"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def encode(data):
  return tokenizer(data['text'],
                   truncation=True,
                   padding='max_length',
                   max_length=32)


data_dir = 'data/banking77/banking_data'
banking_data = Banking77.load(data_dir=data_dir)
train_datasets = banking_data['train']
train_datasets = train_datasets.map(encode, batched=True)
train_datasets = DataLoader(train_datasets,
                            batch_size=16,
                            shuffle=True,
                            drop_last=True,
                            num_workers=2).dataset

val_datasets = banking_data['validation']
val_datasets = val_datasets.map(encode, batched=True)

val_datasets = DataLoader(val_datasets,
                          batch_size=16,
                          shuffle=False,
                          drop_last=True,
                          num_workers=2).dataset

dev_datasets = banking_data['test']
dev_datasets = dev_datasets.map(encode, batched=True)

dev_datasets = DataLoader(dev_datasets,
                          batch_size=16,
                          shuffle=False,
                          drop_last=True,
                          num_workers=2).dataset

label_list = banking_data['train'].features['label'].names
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list), output_hidden_states=True)
banking_datasets = {}
banking_datasets['train'] = train_datasets
banking_datasets['val'] = val_datasets
banking_datasets['test'] = dev_datasets

is_grad = 'grad_'
# for name, param in model.named_parameters():
#   if 'classifier' not in name:  # classifier layer
#     param.requires_grad = False

from sklearn.metrics import accuracy_score


def compute_metrics(pred):
  labels = pred.label_ids
  # preds = pred.predictions.argmax(-1)
  preds = [x.argmax(-1) for x in pred.predictions[0]]
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


batch_size = 32
exprmt_ds = 'banking77'
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
trainer = Trainer(model.to(device),
                  args,
                  train_dataset=banking_datasets['train'],
                  eval_dataset=banking_datasets['val'],
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(args.output_dir)

predictions, labels, _ = trainer.predict(banking_datasets['test'])
# predictions = np.argmax(predictions, axis=1)
predictions = np.argmax(predictions[0], axis=1)
results = accuracy_score(labels, predictions)
print(results)
