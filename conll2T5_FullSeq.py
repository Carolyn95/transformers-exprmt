# Script for turning conll03 data to t5 acceptable format
#source: He said the lifestyle associated with being Miss Universe could make routine exercise difficult .
#target: He O said O the O lifestyle O associated O with O being O Miss B-MISC Universe I-MISC could O make O routine O exercise O difficult O . O
# 20201210
import pdb
import pandas as pd
import os


def convertConllToT5(file_path):
  file_pdir = file_path.split('/')[0]
  # print(file_dir)
  if 'dev' in file_path:
    save_dir = file_pdir + '/processed_conll/val/'
  elif 'test' in file_path:
    save_dir = file_pdir + '/processed_conll/test/'
  elif 'train' in file_path:
    save_dir = file_pdir + '/processed_conll/train/'
  else:
    raise ValueError(
        'Invalid file_path to convert, must be one of `dev`, `train` or `test`')

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  with open(file_path, 'r') as f:
    data = f.read()
    data = data.replace('-DOCSTART- -X- -X- O\n\n', '').strip()
  data = data.split('\n\n')

  sources = []
  targets = []
  for d in data:
    temp_source = [t.split(' ')[0] for t in d.splitlines()]
    temp_source = ' '.join(temp_source)
    sources.append(temp_source)
    temp_target = [
        t.split(' ')[0] + ' ' + t.split(' ')[-1] for t in d.splitlines()
    ]
    temp_target = ' '.join(temp_target)
    targets.append(temp_target)
    # print(temp_source, temp_target)
  assert len(sources) == len(
      targets), 'Sources length does not match Targets length'

  for i, s in enumerate(sources):
    with open(save_dir + str(i) + '.source', 'w') as f:
      f.write(s)
      f.write('\n')
  for i, s in enumerate(targets):
    with open(save_dir + str(i) + '.target', 'w') as f:
      f.write(s)
      f.write('\n')


if __name__ == '__main__':
  convertConllToT5('data/conll_03/dev.txt')
  convertConllToT5('data/conll_03/train.txt')
  convertConllToT5('data/conll_03/test.txt')

  # pdb.set_trace()

  # # temp = d.split()[0] for d in data

  # col1 = [data[i] for i in range(0, len(data), 2)]
  # col2 = [data[i] for i in range(1, len(data), 2)]

  # # Create the data frame
  # df = pd.DataFrame({'Time': col1, 'Text': col2})
  # print(df)
  # pdb.set_trace()

  # df = pd.read_csv('test_s.txt',
  #                 sep=" ",
  #                 skiprows=1,
  #                 header=None,
  #                 names=["word", "pos", "sem", "ner"])
  # for row in df:
  #   source = 1
  #   target = 2
  # pdb.set_trace()

  # with open('test_s.txt', 'r') as f:
  #   data = f.read()
  #   data = data.replace('-DOCSTART- -X- -X- O\n\n', '')

  # pdb.set_trace()
