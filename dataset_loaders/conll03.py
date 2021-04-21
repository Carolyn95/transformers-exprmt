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
"""CONLL03 dataset for Token Classification.   
   Link: https://www.clips.uantwerpen.be/conll2003/ner/
"""
import json
from pathlib import Path
import datasets
import pdb

_CITATION = """CITATION IF ANY"""

_DESCRIPTION = """DESCRIPTION IF ANY"""

_HOMEPAGE = """HOMEPAGE IF ANY"""


class Conll03(datasets.GeneratorBasedBuilder):
  """Conll03 dataset."""

  @staticmethod
  def load(data_dir='data/conll_03', **kwargs):
    src_file = Path(__file__)
    if not data_dir:
      data_dir = src_file.with_suffix('')
    with open(Path(data_dir).joinpath('postags.json')) as fp:
      pos_tags = json.load(fp)
    with open(Path(data_dir).joinpath('chunktags.json')) as fp:
      chunk_tags = json.load(fp)
    with open(Path(data_dir).joinpath('nertags.json')) as fp:
      ner_tags = json.load(fp)

    features = datasets.Features({
        'id': datasets.Value('string'),
        'token': datasets.Value('string'),
        'ner_tag': datasets.features.ClassLabel(names=ner_tags),
        'pos_tag': datasets.features.ClassLabel(names=pos_tags),
        'chunk_tag': datasets.features.ClassLabel(names=chunk_tags)
    })
    return datasets.load_dataset(
        str(src_file.absolute()),
        data_dir=data_dir,
        features=features,
        **kwargs)  # data_dir here can be referenced from `self.config.data_dir`

  def _info(self):
    return datasets.DatasetInfo(citation=_CITATION,
                                description=_DESCRIPTION,
                                homepage=_HOMEPAGE)

  def _split_generators(self, dl_manager):
    data = dl_manager.download_and_extract({
        split: Path(self.config.data_dir).joinpath(f'{split}.txt')
        for split in ['train', 'dev', 'test']
    })

    return [
        datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                gen_kwargs={'datadir': data['train']}),
        datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                gen_kwargs={'datadir': data['dev']}),
        datasets.SplitGenerator(name=datasets.Split.TEST,
                                gen_kwargs={'datadir': data['test']})
    ]

  def _generate_examples(self, datadir):
    with open(datadir) as fp:
      samples = fp.read().splitlines()
    for i, eg in enumerate(samples):
      if "-DOCSTART-" in eg or eg == "" or eg == "\n":
        continue
      else:
        eg = eg.split(' ')
        yield i, {
            'id': str(i),
            'token': eg[0],
            'pos_tag': eg[1],
            'chunk_tag': eg[2],
            'ner_tag': eg[3]
        }
