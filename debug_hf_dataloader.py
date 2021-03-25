import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pdb
import datasets
from datasets import load_metric, load_dataset
import random
import pandas as pd
from transformers import AutoTokenizer
import transformers
# from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
# from transformers import DataCollatorForTokenClassification
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def loadDataset(data_file):
  # composed synthetic data into one txt file and read string into sents list and labels list respectively
  sents = []
  labels = []
  with open(data_file) as f:
    data = f.read().strip().split('\n\n')
    for d in data:
      word_label_pairs = d.split('\n')
      temp_sent = []
      temp_label = []
      for word_label_pair in word_label_pairs:
        word, label = word_label_pair.split(' ')
        temp_sent.append(word)
        temp_label.append(label)
      sents.append(' '.join(temp_sent))
      labels.append(' '.join(temp_label))
  random.shuffle(sents)
  random.shuffle(labels)
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
    self.dataset_sents, self.dataset_labels = loadDataset(data_file_path)
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

    test_dataset_sents = self.dataset_sents[subset_len:]
    test_dataset_labels = self.dataset_labels[subset_len:]
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


if __name__ == '__main__':
  data_dir = './data/synthetic-data/new_app_device_loc.txt'
  all_datasets = getDataset(data_dir=data_dir, is_shuffle=True)
  pdb.set_trace()
  print()
