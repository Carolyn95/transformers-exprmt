# keep the original format
# subset 1500 of trainning dataset for both conll03 and gmb
# data/conll_03/train.txt
# data/gmb/text_train.txt
# data/gmb/labels_train.txt

import pdb
subset_number = 1500
# conll
with open('data/conll_03/train.txt', 'r') as f:
  data = f.read()
  data = data.replace('-DOCSTART- -X- -X- O\n\n', '').strip()
conll_data = data.split('\n\n')
sub_conll = conll_data[:subset_number]

f = open('temp/conll_1500.txt', 'w')
for d in sub_conll:
  f.write(d)
  f.write('\n\n')
f.close()

# gmb
with open('data/gmb/text_train.txt', 'r') as f:
  data = f.read()
gmb_data = data.split('\n')
sub_gmb_data = gmb_data[:subset_number]
f = open('temp/gmb_text_train_1500.txt', 'w')
for d in sub_gmb_data:
  f.write(d)
  f.write('\n')
f.close()

with open('data/gmb/labels_train.txt', 'r') as f:
  data = f.read()
gmb_labels = data.split('\n')
sub_gmb_labels = gmb_labels[:subset_number]

f = open('temp/gmb_labels_train_1500.txt', 'w')
for d in sub_gmb_labels:
  f.write(d)
  f.write('\n')
f.close()
