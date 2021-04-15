# coding=utf-8
# Copyright Carolyn CHEN
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BANKING77 dataset for Intent Detection. 
   Please visit PolyAI github page to obtain data and authorization.     
"""
import datasets
from pathlib import Path
import csv
import pdb
import json
from collections import Counter
from random import shuffle

_CITATION = """CITATION IF ANY"""

_DESCRIPTION = """DESCRIPTION IF ANY"""

_HOMEPAGE = """HOMEPAGE IF ANY"""


class BankingAll(datasets.GeneratorBasedBuilder):
  """Banking77 dataset from its homepage comes with 'categories.json', 'test.csv', 'train.csv'
     'categoties.json' contains labels of the dataset
     'test.csv' and 'train.csv' are the two splits of the dataset
     class method will separate the training set into 'train' and 'evaliation'
  """

  @staticmethod
  def load(data_dir='data/banking77/banking_data', **kwargs):
    """Returns DatasetDict."""
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
    """Defines dataset info."""
    return datasets.DatasetInfo(description=_DESCRIPTION,
                                homepage=_HOMEPAGE,
                                citation=_CITATION)

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    data = dl_manager.download_and_extract({
        split: Path(self.config.data_dir).joinpath(f'{split}.csv')
        for split in ['train', 'test']
    })
    with open(data['train']) as fp:
      all_train = [x for x in csv.DictReader(fp)]
    with open(data['test']) as fp:
      test = [x for x in csv.DictReader(fp)]
    # Extract balanced validation split from training data
    val_split = []
    labels = [x['category'] for x in all_train]
    for label, count in Counter(labels).items():
      n = count // 5  # Take ~(1/5) from training as validation
      val_split.extend([i for i, c in enumerate(labels) if c == label][:n])
    train_split = [x for x in range(len(labels)) if x not in val_split]
    return [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={'examples': [all_train[i] for i in train_split]}),
        datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={'examples': [all_train[i] for i in val_split]}),
        datasets.SplitGenerator(name=datasets.Split.TEST,
                                gen_kwargs={'examples': test})
    ]

  def _generate_examples(self, examples):
    """Yield examples."""
    for i, eg in enumerate(examples):
      yield i, {'id': str(i), 'text': eg['text'], 'label': eg['category']}
