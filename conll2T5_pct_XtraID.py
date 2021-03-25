# Script for turning conll03 data to t5 acceptable format
# input:	<extra_id_0>SOCCER <extra_id_1>- <extra_id_2>JAPAN <extra_id_3>GET <extra_id_4>LUCKY <extra_id_5>WIN <extra_id_6>, <extra_id_7>CHINA <extra_id_8>IN <extra_id_9>SURPRISE <extra_id_10>DEFEAT <extra_id_11>.
# target:	O<extra_id_0>  O<extra_id_1>  I-LOC<extra_id_2>  O<extra_id_3>  O<extra_id_4>  O<extra_id_5>  O<extra_id_6>  I-PER<extra_id_7>  O<extra_id_8>  O<extra_id_9>  O<extra_id_10>  O<extra_id_11>
# # 20201219
# original: 14986
# 10%: 1500
# 1%: 150
# 0.5%: 75
import pdb
import pandas as pd
import os
# import nltk


def convertConllToT5(file_path, topk=1500):
  file_pdir = file_path.split('/')[0]
  # print(file_dir)
  if 'dev' in file_path:
    save_dir = file_pdir + '/processed_' + str(topk) + '_conll_XtraID/val/'
    with open(file_path, 'r') as f:
      data = f.read()
      data = data.replace('-DOCSTART- -X- -X- O\n\n', '').strip()
      data = data.split('\n\n')
  elif 'test' in file_path:
    save_dir = file_pdir + '/processed_' + str(topk) + '_conll_XtraID/test/'
    with open(file_path, 'r') as f:
      data = f.read()
      data = data.replace('-DOCSTART- -X- -X- O\n\n', '').strip()
      data = data.split('\n\n')
  elif 'train' in file_path:
    save_dir = file_pdir + '/processed_' + str(topk) + '_conll_XtraID/train/'
    with open(file_path, 'r') as f:
      data = f.read()
      data = data.replace('-DOCSTART- -X- -X- O\n\n', '').strip()
      data = data.split('\n\n')[:topk]
  else:
    raise ValueError(
        'Invalid file_path to convert, must be one of `dev`, `train` or `test`')

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  sources = []
  targets = []
  for d in data:
    temp_source = [t.split(' ')[0] for t in d.splitlines()]
    temp_target = [t.split(' ')[-1] for t in d.splitlines()]
    # tokens = nltk.word_tokenize(temp_source[0])
    assert len(temp_source) == len(
        temp_target), 'Sources length does not match Targets length'
    seq_len = len(temp_target)
    source_xtra, target_xtra = [], []
    for source_tk, target_tk, idx in zip(temp_source, temp_target,
                                         range(seq_len)):
      xtra_tk = '<extra_id_' + str(idx) + '>'
      source_xtra.append(source_tk.strip())
      source_xtra.append(xtra_tk)
      target_xtra.append(target_tk.strip())
      target_xtra.append(xtra_tk)

    temp_source = ' '.join(source_xtra)
    sources.append(temp_source)
    temp_target = ' '.join(target_xtra)
    targets.append(temp_target)
    # print(temp_source, temp_target)

  for i, s in enumerate(sources):
    with open(save_dir + str(i) + '.source', 'w') as f:
      f.write(s)
      f.write('\n')
  for i, s in enumerate(targets):
    with open(save_dir + str(i) + '.target', 'w') as f:
      f.write(s)
      f.write('\n')


if __name__ == '__main__':
  topk = 300
  convertConllToT5('data/conll_03/train.txt', topk=topk)
  convertConllToT5('data/conll_03/dev.txt', topk=topk)
  convertConllToT5('data/conll_03/test.txt', topk=topk)
