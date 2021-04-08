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
import csv
import json
from pathlib import Path


class Banking77(datasets.GeneratorBasedBuilder):

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

  def _info(self):
    return datasets.info.DatasetInfo(supervised_keys=None)

  def _split_generators(self, dl_manager):
    data = dl_manager.download_and_extract({
        split: Path(self.config.data_dir).joinpath(f'{split}.csv')
        for split in ['train', 'test']
    })
    return [
        datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                gen_kwargs={'filepath': data['train']}),
        datasets.SplitGenerator(name=datasets.Split.TEST,
                                gen_kwargs={'filepath': data['test']})
    ]

  def _generate_examples(self, filepath):
    with open(filepath) as fp:
      samples = [x for x in csv.DictReader(fp)]
    for i, eg in enumerate(samples):
      yield i, {'id': str(i), 'text': eg['text'], 'label': eg['category']}


# "distilbert-base-uncased" | "distilbert-base-cased"
# "bert-base-uncased" | "bert-base-cased"

model_checkpoint = "bert-base-uncased"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_dir = 'data/banking77/banking_data'
banking_data = Banking77().load(data_dir=data_dir)
# paddding_max_length = 32
pdb.set_trace()
print()