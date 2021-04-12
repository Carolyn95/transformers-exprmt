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
"""CLINC150 dataset for Intent Detection. 
   Please visit PolyAI github page to obtain data and authorization. 
   Philosophy:
      REFERENCE: https://huggingface.co/docs/datasets/_images/datasets_doc.jpg
      Create a new class inherits `datasets.GeneratorBasedBuilder` class.
      This sub-class needs to override 3 methods:
      {
        `_info`: descriptive information of the dataset, including citation, etc.
                 You could define features used in the dataset here also, 
                 features is an object of `datasets.Features`

        `_split_generator`: returns generator of each split of the dataset

        `_generate_examples`: yield dataset example in each split consuming by DatasetLoader
      }
      
"""

import csv
import json
from pathlib import Path
import datasets
import pdb

_CITATION = """CITATION IF ANY"""

_DESCRIPTION = """DESCRIPTION IF ANY"""

_HOMEPAGE = """HOMEPAGE IF ANY"""


class ClincAll(datasets.GeneratorBasedBuilder):
  """Clinc150 dataset from its home page comes with 'categories.json', 'train.csv', 'val.csv', 'test.csv'
     categories.json contains labels of the dataset
     test.csv, val.csv, train.csv are the three splits of this dataset
  """

  @staticmethod
  def load(data_dir='data/clinc150', **kwargs):
    """Returns DataDict."""
    src_file = Path(__file__)  # get directory of this src file
    if not data_dir:
      data_dir = src_file.with_suffix('')
    with open(Path(data_dir).joinpath('categories.json')) as fp:
      features = datasets.Features({  # features and datatype of each column
          'id': datasets.Value('string'),  # alternative
          'text': datasets.Value('string'),
          'label': datasets.features.ClassLabel(names=json.load(fp))
      })
    return datasets.load_dataset(
        str(src_file.absolute()),
        data_dir=data_dir,  # argument of class config
        features=features,
        **kwargs)

  def _info(self):
    """Defines dataset info."""
    return datasets.DatasetInfo(citation=_CITATION,
                                homepage=_HOMEPAGE,
                                description=_DESCRIPTION)

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    data = dl_manager.download_and_extract({
        split: Path(self.config.data_dir).joinpath(f'{split}.csv')
        for split in ['train', 'val', 'test']
    })
    return [
        datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                gen_kwargs={
                                    'filepath': data['train']
                                }),  # kw defined in _generate_example
        datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                gen_kwargs={'filepath': data['val']}),
        datasets.SplitGenerator(name=datasets.Split.TEST,
                                gen_kwargs={'filepath': data['test']})
    ]

  def _generate_examples(self, filepath):
    """Yield examples."""
    with open(filepath) as fp:
      samples = [x for x in csv.DictReader(fp)]
    for i, eg in enumerate(samples):
      # yield example should match dataset features as defined
      yield i, {'id': str(i), 'text': eg['text'], 'label': eg['category']}
