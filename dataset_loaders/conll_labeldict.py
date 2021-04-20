"""
Pre-requisites for conll dataloader. 
This script is for saving `pos_tags`, `chunk_tags` and `ner_tags` from conll03 dataset.
The downloaded conll03 dataset contains 3 files 'train.txt', 'dev.txt' and 'test.txt'.
Ouput of this script is save 3 files in the same directory of downloaded conll dataset.
Saved files can be used to build features for conll03 DataDict.
Invalid lines(eg. new lines only, empty lines, doc starts) ratio: 0.0725 = 15931/219552.
`POS_TAGS_REF`, `CHUNK_TAGS_REF` and `NER_TAGS_REF` are from: https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py

"""
import csv
import pdb
import json
POS_TAGS_REF = [
    '"',
    "''",
    "#",
    "$",
    "(",
    ")",
    ",",
    ".",
    ":",
    "``",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "NN|SYM",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
]
CHUNK_TAGS_REF = [
    "O",
    "B-ADJP",
    "I-ADJP",
    "B-ADVP",
    "I-ADVP",
    "B-CONJP",
    "I-CONJP",
    "B-INTJ",
    "I-INTJ",
    "B-LST",
    "I-LST",
    "B-NP",
    "I-NP",
    "B-PP",
    "I-PP",
    "B-PRT",
    "I-PRT",
    "B-SBAR",
    "I-SBAR",
    "B-UCP",
    "I-UCP",
    "B-VP",
    "I-VP",
]
NER_TAGS_REF = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def main(conll_train_path):

  tokens = []
  pos_tags = []
  chunk_tags = []
  ner_tags = []
  problems = []

  with open(conll_train, 'r') as fp:
    data = fp.read().splitlines()

    print(len(data))
    for x in data:
      if "-DOCSTART-" in x or x == "" or x == "\n":
        problems.append(data.index(x))
      else:
        splits = x.split(' ')
        tokens.append(splits[0])
        pos_tags.append(splits[1])
        chunk_tags.append(splits[2])
        ner_tags.append(splits[3])

  pos_tags = list(set(pos_tags))
  chunk_tags = list(set(chunk_tags))
  ner_tags = list(set(ner_tags))

  temp = [x for x in POS_TAGS_REF if x not in pos_tags]
  print(
      f'There are {len(temp)} pos_tags in standard conll not in downloaded files, possible file compromising alert.'
  )
  print(f'They are {temp}\n')

  temp = [x for x in POS_TAGS_REF if x not in chunk_tags]
  print(
      f'There are {len(temp)} chunk_tags in standard conll not in downloaded files, possible file compromising alert.'
  )
  print(f'They are {temp}\n')

  temp = [x for x in NER_TAGS_REF if x not in ner_tags]
  print(
      f'There are {len(temp)} ner_tags in standard conll not in downloaded files, possible file compromising alert.'
  )
  print(f'They are {temp}\n')

  with open("../data/conll_03/postags.json", "w") as f:
    f.write(json.dumps(pos_tags))

  with open("../data/conll_03/chunktags.json", "w") as f:
    f.write(json.dumps(chunk_tags))

  with open("../data/conll_03/nertags.json", "w") as f:
    f.write(json.dumps(ner_tags))


if __name__ == '__main__':
  conll_train = "../data/conll_03/train.txt"
  main(conll_train)
