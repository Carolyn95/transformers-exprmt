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


if __name__ == '__main__':
  data_dir = './data/gmb/'
  train_datasets = getDataset(data_dir=data_dir,
                              type_path='train',
                              is_shuffle=True)
  dev_datasets = getDataset(data_dir=data_dir,
                            type_path='dev',
                            is_shuffle=False)
  pdb.set_trace()
  print()