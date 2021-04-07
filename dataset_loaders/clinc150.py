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
"""CLINC150 dataset used by PolyAI for Intent Detection."""

import csv
import json
from pathlib import Path

import datasets

_CITATION = """
@inproceedings{larson-etal-2019-evaluation,
title = {An Evaluation Dataset for Intent Classification and Out-of-Scope
         Prediction},
author = {Larson, Stefan and
      Mahendran, Anish  and
      Peper, Joseph J.  and
      Clarke, Christopher  and
      Lee, Andrew  and
      Hill, Parker  and
      Kummerfeld, Jonathan K.  and
      Leach, Kevin  and
      Laurenzano, Michael A.  and
      Tang, Lingjia  and
      Mars, Jason},
booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural
             Language Processing and the 9th International Joint Conference on
             Natural Language Processing (EMNLP-IJCNLP)},
year = {2019},
url = {https://www.aclweb.org/anthology/D19-1131}
}
"""

_DESCRIPTION = """
We introduce a new crowdsourced dataset of 23,700 queries,including 22,500
in-scope queries covering 150 intents, which can be grouped into 10 general
domains. The dataset also includes 1,200 out-of-scope queries.
"""

_HOMEPAGE = 'https://github.com/clinc/oos-eval'


class Clinc150(datasets.GeneratorBasedBuilder):
  """CLINC150 dataset used by PolyAI for Intent Detection."""

  @staticmethod
  def load(data_dir=None, **kwargs):
    """Returns DatasetDict.

        For convenience, create a symbolic link to the dataset folder here with
        the same name as this source file (ie. "./clinc150").

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
