# Script for turning conll03 data to t5 acceptable format
# input:	<extra_id_0>SOCCER <extra_id_1>- <extra_id_2>JAPAN <extra_id_3>GET <extra_id_4>LUCKY <extra_id_5>WIN <extra_id_6>, <extra_id_7>CHINA <extra_id_8>IN <extra_id_9>SURPRISE <extra_id_10>DEFEAT <extra_id_11>.
# target:	O<extra_id_0>  O<extra_id_1>  I-LOC<extra_id_2>  O<extra_id_3>  O<extra_id_4>  O<extra_id_5>  O<extra_id_6>  I-PER<extra_id_7>  O<extra_id_8>  O<extra_id_9>  O<extra_id_10>  O<extra_id_11>
# # 20210203
import pdb
import pandas as pd
import os
# import nltk
# validation set == training set, due to the nature of data


def convertGmbToT5(text_file_path, label_file_path):

  with open(text_file_path, 'r') as f:
    sents = f.read().split('\n')

  with open(label_file_path, 'r') as f:
    labels = f.read().split('\n')

  for one_sent, sent_label in zip(sents, labels):
    source = []
    target = []
    one_sent_tks = one_sent.split(' ')
    sent_label_tks = sent_label.split(' ')
    length = len(one_sent_tks)
    for sent_tk, label_tk, idx in zip(one_sent_tks, sent_label_tks,
                                      range(length)):
      xtra_tk = '<extra_id_' + str(idx) + '>'
      source.append(sent_tk.strip())
      source.append(xtra_tk)
      target.append(label_tk.strip())
      target.append(xtra_tk)
    source = ' '.join(source)
    target = ' '.join(target)
    pdb.set_trace()
    print(source, target)


if __name__ == '__main__':
  text_file_path = 'data/gmb/text_dev.txt'
  label_file_path = 'data/gmb/labels_dev.txt'

  convertGmbToT5(text_file_path, label_file_path)
