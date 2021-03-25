import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk

from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)
import torch
import os
import shutil
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pdb
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer

model_checkpoint = "bert-base-uncased"
labels = 1
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=labels, output_hidden_states=True)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_len = 32
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)
outputs = model(**inputs, labels=labels)
pdb.set_trace()
loss = outputs.loss
logits = outputs.logits
