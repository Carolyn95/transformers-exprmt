"""
20210225
funnel transformer
gmb dataset
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import pdb
import datasets
from datasets import load_metric, load_dataset
import random
import pandas as pd
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GmbData(Dataset):

  def __init__(self, data_dir, type_path, max_len=32):
    self.data_dir = data_dir
    self.type_path = type_path
    self.sents_file_path = data_dir + 'text_' + type_path + '.txt'
    self.label_file_path = data_dir + 'labels_' + type_path + '.txt'
    self.max_len = max_len
    self._build()

  def __getitem__(self, index):
    return self.datasets[index]

  def __len__(self):
    return len(self.dataset_sents)

  def _build(self):
    self._buil_examples_from_files(self.sents_file_path, self.label_file_path)

  def _buil_examples_from_files(self, sents_file_path, label_file_path):
    self.dataset_sents = load_dataset('text',
                                      data_files={
                                          self.type_path: self.sents_file_path
                                      })[self.type_path]['text']
    self.dataset_labels = load_dataset('text',
                                       data_files={
                                           self.type_path: self.label_file_path
                                       })[self.type_path]['text']
    tags = list(set([t for dl in self.dataset_labels for t in dl.split()]))
    self.ner_tags = list(sorted(tags))
    self.datasets = []
    for idx, (sent_line, label_line) in enumerate(
        zip(self.dataset_sents, self.dataset_labels)):
      temp = {}
      temp['id'] = idx
      temp['tokens'] = sent_line.split(' ')
      # temp['ner_tags'] = label_line.split(' ')
      temp['ner_tags'] = [
          i for s in label_line.split(" ")
          for (i, l) in enumerate(self.ner_tags) if l == s
      ]
      self.datasets.append(temp)

  def select(self, subset_len):
    self._buil_subset_examples_from_self(subset_len)

  def _buil_subset_examples_from_self(self, subset_len):
    self.dataset_sents = self.dataset_sents[:subset_len]
    self.dataset_labels = self.dataset_labels[:subset_len]
    self.datasets = []
    for idx, (sent_line, label_line) in enumerate(
        zip(self.dataset_sents, self.dataset_labels)):
      temp = {}
      temp['id'] = idx
      temp['tokens'] = sent_line.split(' ')
      # temp['ner_tags'] = label_line.split(' ')
      temp['ner_tags'] = [
          i for s in label_line.split(" ")
          for (i, l) in enumerate(self.ner_tags) if l == s
      ]
      self.datasets.append(temp)


def getDataset(data_dir, type_path, is_shuffle):
  dataset = GmbData(data_dir=data_dir, type_path=type_path)
  dataset_loader = DataLoader(dataset,
                              batch_size=16,
                              shuffle=is_shuffle,
                              drop_last=True,
                              num_workers=2).dataset
  return dataset_loader


# "funnel-transformer/small-base",
# "funnel-transformer/intermediate", "funnel-transformer/xlarge-base"
# "funnel-transformer/medium-base", "funnel-transformer/large-base"

model_checkpoint = "funnel-transformer/xlarge"
model_name = "funnel-xlarge"

data_dir = './data/gmb/'

train_datasets = getDataset(data_dir=data_dir,
                            type_path='train',
                            is_shuffle=True)
dev_datasets = getDataset(data_dir=data_dir, type_path='dev', is_shuffle=False)

label_list = train_datasets.ner_tags

# 75, 150, 300, 1500, len(train_datasets)
traning_samples = len(train_datasets)
train_datasets.select(traning_samples)

datasets = {}
datasets['train'] = train_datasets
datasets['test'] = dev_datasets

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
label_all_tokens = True


def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples["tokens"],
                               truncation=True,
                               is_split_into_words=True)

  labels = []
  word_ids = tokenized_inputs.word_ids(batch_index=0)
  for word_idx in word_ids:
    previous_word_idx = None
    for label in examples["ner_tags"]:
      # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
      if word_idx is None:
        labels.append(-100)
      # We set the label for the first token of each word.
      elif word_idx != previous_word_idx:
        labels.append(label)
      # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
      else:
        labels.append(label if label_all_tokens else -100)
      previous_word_idx = word_idx
      break
  tokenized_inputs["labels"] = labels
  return tokenized_inputs


tokenized_datasets_train = list(
    map(lambda x: tokenize_and_align_labels(x), datasets['train']))

tokenized_datasets_test = list(
    map(lambda x: tokenize_and_align_labels(x), datasets['test']))
tokenized_datasets = {}
tokenized_datasets['train'] = tokenized_datasets_train
tokenized_datasets['test'] = tokenized_datasets_test

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list))

is_grad = 'grad_'
# for name, param in model.named_parameters():
#   if 'classifier' not in name:  # classifier layer
#     param.requires_grad = False

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric('seqeval')


def compute_metrics(p):
  predictions, labels = p
  predictions = np.argmax(predictions, axis=2)
  true_predictions = [[
      label_list[p] for (p, l) in zip(prediction, label) if l != -100
  ] for prediction, label in zip(predictions, labels)]
  true_labels = [[
      label_list[l] for (p, l) in zip(prediction, label) if l != -100
  ] for prediction, label in zip(predictions, labels)]
  results = metric.compute(predictions=true_predictions, references=true_labels)
  return {
      'precision': results['overall_precision'],
      'recall': results['overall_recall'],
      'f1': results['overall_f1'],
      'accuracy': results['overall_accuracy']
  }


batch_size = 8
exprmt_ds = 'gmb'
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
                  train_dataset=tokenized_datasets['train'],
                  eval_dataset=tokenized_datasets['train'],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(args.output_dir)
# eval_result = trainer.evaluate()
# print(eval_result)

predictions, labels, _ = trainer.predict(tokenized_datasets['test'])
predictions = np.argmax(predictions, axis=2)
true_predictions = [[
    label_list[p] for (p, l) in zip(prediction, label) if l != -100
] for prediction, label in zip(predictions, labels)]
true_labels = [[
    label_list[l] for (p, l) in zip(prediction, label) if l != -100
] for prediction, label in zip(predictions, labels)]
results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)
