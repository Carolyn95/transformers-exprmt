"""
20210301
funnel transformer
wikiner dataset (download asset from spacy pipeline) => `~/Projects/mygit/huggingface-exprmt/data/wikiner`

process conll-like dataset into gmp-like and 'reuse' gmp data processing pipeline

original dataset `aij-wikiner-en-wp2` is like:
The|DT|I-MISC Oxford|NNP|I-MISC Companion|NNP|I-MISC to|TO|I-MISC Philosophy|NNP|I-MISC says|VBZ|O ,|,|O "|LQU|O there|EX|O is|VBZ|O no|DT|O single|JJ|O defining|VBG|O position|NN|O that|IN|O all|DT|O anarchists|NNS|O hold|VBP|O ,|,|O and|CC|O those|DT|O considered|VBN|O anarchists|NNS|O at|IN|O best|JJS|O share|NN|O a|DT|O certain|JJ|O family|NN|O resemblance|NN|O .|.|O "|RQU|O
                    |
                    v
new line character '\n' to separate line
space ' ' to separate word 
verticle line '|' to separate word and label

e.g)
The|DT|I-MISC Oxford|NNP|I-MISC Companion|NNP|I-MISC to|TO|I-MISC Philosophy|NNP|I-MISC says|VBZ|O ,|,|O "|LQU|O there|EX|O is|VBZ|O no|DT|O single|JJ|O defining|VBG|O position|NN|O that|IN|O all|DT|O anarchists|NNS|O hold|VBP|O ,|,|O and|CC|O those|DT|O considered|VBN|O anarchists|NNS|O at|IN|O best|JJS|O share|NN|O a|DT|O certain|JJ|O family|NN|O resemblance|NN|O .|.|O "|RQU|O
In|IN|O the|DT|O end|NN|O ,|,|O for|IN|O anarchist|JJ|O historian|JJ|O Daniel|NNP|I-PER Guerin|NNP|I-PER "|LQU|O Some|DT|O anarchists|NNS|O are|VBP|O more|RBR|O individualistic|JJ|O than|IN|O social|JJ|O ,|,|O some|DT|O more|JJR|O social|JJ|O than|IN|O individualistic|JJ|O .|.|O

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pdb
import datasets
from datasets import load_metric
import random
import pandas as pd
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_dataset(data_file):
  sents = []
  labels = []
  with open(data_file) as f:
    data = f.read().strip().split('\n')
    for d in data:
      word_label_pairs = d.strip().split(' ')
      temp_sent = []
      temp_label = []
      for word_label_pair in word_label_pairs:
        try:
          word, _, label = word_label_pair.strip().split('|')
        except:
          continue
        temp_sent.append(word)
        temp_label.append(label)
      sents.append(' '.join(temp_sent))
      labels.append(' '.join(temp_label))
  sent_label_pairs = list(zip(sents, labels))
  random.shuffle(sent_label_pairs)
  sents, labels = zip(*sent_label_pairs)
  print('sents length is {}, labels length is {}'.format(
      len(sents), len(labels)))

  return sents, labels


class SyntheticData(Dataset):

  def __init__(self, data_dir, max_len=32):
    self.data_file_path = data_dir
    self.max_len = max_len
    self._build()

  def __getitem__(self, index):
    return self.datasets[index]

  def __len__(self):
    return len(self.dataset_sents)

  def _build(self):
    self._buil_examples_from_files(self.data_file_path)

  def _buil_examples_from_files(self, data_file_path):
    self.dataset_sents, self.dataset_labels = load_dataset(data_file_path)
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
    train_datasets, test_datasets = self._buil_subset_examples_from_self(
        subset_len)
    return train_datasets, test_datasets

  def _buil_subset_examples_from_self(self, subset_len):
    train_dataset_sents = self.dataset_sents[:subset_len]
    train_dataset_labels = self.dataset_labels[:subset_len]
    train_datasets = []

    for idx, (sent_line, label_line) in enumerate(
        zip(train_dataset_sents, train_dataset_labels)):
      temp = {}
      temp['id'] = idx
      temp['tokens'] = sent_line.split(' ')
      # temp['ner_tags'] = label_line.split(' ')
      temp['ner_tags'] = [
          i for s in label_line.split(" ")
          for (i, l) in enumerate(self.ner_tags) if l == s
      ]
      train_datasets.append(temp)

    # test_dataset_sents = self.dataset_sents[subset_len:]
    test_dataset_sents = self.dataset_sents[-3000:]
    # test_dataset_labels = self.dataset_labels[subset_len:]
    test_dataset_labels = self.dataset_labels[-3000:]
    test_datasets = []
    for idx, (sent_line, label_line) in enumerate(
        zip(test_dataset_sents, test_dataset_labels)):
      temp = {}
      temp['id'] = idx
      temp['tokens'] = sent_line.split(' ')
      # temp['ner_tags'] = label_line.split(' ')
      temp['ner_tags'] = [
          i for s in label_line.split(" ")
          for (i, l) in enumerate(self.ner_tags) if l == s
      ]
      test_datasets.append(temp)
    return train_datasets, test_datasets


def getDataset(data_dir, is_shuffle):
  dataset = SyntheticData(data_dir=data_dir)
  dataset_loader = DataLoader(dataset,
                              batch_size=16,
                              shuffle=is_shuffle,
                              drop_last=True,
                              num_workers=2).dataset
  return dataset_loader


# "funnel-transformer/small-base" | "funnel-transformer/intermediate" | "funnel-transformer/xlarge-base" | "funnel-transformer/medium-base" | "funnel-transformer/large-base"
# "distilbert-base-uncased" | "distilbert-base-cased" | "bert-base-uncased" | "bert-base-cased"

model_checkpoint = "bert-base-uncased"
model_name = "bert-base-uncased"

data_dir = 'data/wikiner/aij-wikiner-en-wp2'

all_datasets = getDataset(data_dir=data_dir, is_shuffle=True)

label_list = all_datasets.ner_tags

# 75, 150, 300, 1500, len(train_datasets)
traning_samples = 75
train_datasets, dev_datasets = all_datasets.select(traning_samples)

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


batch_size = 16
exprmt_ds = 'wikiner'
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
