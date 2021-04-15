# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script
# contributor.
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
"""BANKING 77 dataset used by PolyAI for Intent Detection."""

import csv
import json
from pathlib import Path

import datasets

_CITATION = """
@inproceedings{Casanueva2020,
author = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew
          Henderson and Ivan Vulic},
title = {Efficient Intent Detection with Dual Sentence Encoders},
year = {2020},
month = {mar},
url = {https://arxiv.org/abs/2003.04807},
booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020}
}
"""

_DESCRIPTION = """
Therefore, to complement the recent effort on data collection for intent
detection, we propose a new single-domain dataset: it provides a very
fine-grained set of intents in a banking domain, not present in HWU 64 and
CLINC 150. The new BANKING 77 dataset comprises 13,083 customer service queries
labeled with 77 intents. Its focus on fine-grained single-domain intent
detection makes it complementary to the two other datasets: we believe that any
comprehensive intent detection evaluation should involve both coarser-grained
multi-domain datasets such as HWU 64 and CLINC 150, and a fine-grained
single-domain dataset such as BANKING 77.
"""

_HOMEPAGE = 'https://github.com/PolyAI-LDN/task-specific-datasets'


class Banking77(datasets.GeneratorBasedBuilder):
  """BANKING 77 dataset used by PolyAI for Intent Detection."""

  @staticmethod
  def load(data_dir=None, **kwargs):
    """Returns DatasetDict.

        For convenience, create a symbolic link to the dataset folder here with
        the same name as this source file (ie. "./banking77").

        Args:
            data_dir: folder containing dataset files

        Returns:
            DatasetDict.

        Raises:
            FileNotFoundError if dataset files cannot be found.
        """
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
    return datasets.DatasetInfo(
        description=_DESCRIPTION,
        supervised_keys=None,
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    data = dl_manager.download_and_extract({
        split: Path(self.config.data_dir).joinpath(f'{split}.csv')
        for split in ['train', 'val', 'test']
    })
    return [
        datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                gen_kwargs={'filepath': data['train']}),
        datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                gen_kwargs={'filepath': data['val']}),
        datasets.SplitGenerator(name=datasets.Split.TEST,
                                gen_kwargs={'filepath': data['test']})
    ]

  def _generate_examples(self, filepath):
    """Yields examples."""
    with open(filepath) as fp:
      samples = [x for x in csv.DictReader(fp)]
    for i, eg in enumerate(samples):
      yield i, {'id': str(i), 'text': eg['text'], 'label': eg['category']}
