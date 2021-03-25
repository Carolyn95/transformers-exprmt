# Master script for fine-tuning T5
# !pip install transformers==2.9.0
# !pip install pytorch_lightning==0.7.5

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
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ref from pytorch-lightening: https://github.com/huggingface/transformers/blob/master/examples/lightning_base.py
class T5FineTuner(pl.LightningModule):

  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.hparams = hparams

    self.model = T5ForConditionalGeneration.from_pretrained(
        hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

  def is_logger(self):
    return self.trainer.proc_rank <= 0

  def forward(self,
              input_ids,
              attention_mask=None,
              decoder_input_ids=None,
              decoder_attention_mask=None,
              lm_labels=None):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        lm_labels=lm_labels,
    )

  def _step(self, batch):
    lm_labels = batch["target_ids"]
    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(input_ids=batch["source_ids"],
                   attention_mask=batch["source_mask"],
                   lm_labels=lm_labels,
                   decoder_attention_mask=batch['target_mask'])

    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}

  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss}
    return {
        "avg_train_loss": avg_train_loss,
        "log": tensorboard_logs,
        'progress_bar': tensorboard_logs
    }

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    return {
        "avg_val_loss": avg_loss,
        "log": tensorboard_logs,
        'progress_bar': tensorboard_logs
    }

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=self.hparams.learning_rate,
                      eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]

  def optimizer_step(self,
                     epoch,
                     batch_idx,
                     optimizer,
                     optimizer_idx,
                     second_order_closure=None):
    if self.trainer.use_tpu:
      xm.optimizer_step(optimizer)
    else:
      optimizer.step()
    optimizer.zero_grad()
    self.lr_scheduler.step()

  def get_tqdm_dict(self):
    tqdm_dict = {
        "loss": "{:.3f}".format(self.trainer.avg_loss),
        "lr": self.lr_scheduler.get_last_lr()[-1]
    }

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer,
                                type_path="train",
                                args=self.hparams)
    dataloader = DataLoader(train_dataset,
                            batch_size=self.hparams.train_batch_size,
                            drop_last=True,
                            shuffle=True,
                            num_workers=4)
    t_total = ((len(dataloader.dataset) //
                (self.hparams.train_batch_size * max(1, self.hparams.n_gpu))) //
               self.hparams.gradient_accumulation_steps *
               float(self.hparams.num_train_epochs))
    scheduler = get_linear_schedule_with_warmup(
        self.opt,
        num_warmup_steps=self.hparams.warmup_steps,
        num_training_steps=t_total)
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = get_dataset(tokenizer=self.tokenizer,
                              type_path="val",
                              args=self.hparams)
    return DataLoader(val_dataset,
                      batch_size=self.hparams.eval_batch_size,
                      num_workers=4)


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):

  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir,
                                              "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))


args_dict = dict(
    data_dir="",  # path for data files
    output_dir="",  # path to save the checkpoints
    model_name_or_path='t5-base',
    tokenizer_name_or_path='t5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=
    False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level=
    'O1',  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=
    1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

# IMDB dataset preparation

# !wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xvf aclImdb_v1.tar.gz

train_pos_files = glob.glob('aclImdb/train/pos/*.txt')
train_neg_files = glob.glob('aclImdb/train/neg/*.txt')
len(train_pos_files), len(train_neg_files)
# !mkdir aclImdb/val aclImdb/val/pos aclImdb/val/neg
random.shuffle(train_pos_files)
random.shuffle(train_neg_files)

val_pos_files = train_pos_files[:1000]
val_neg_files = train_neg_files[:1000]

import shutil
for f in val_pos_files:
  shutil.move(f, 'aclImdb/val/pos')
for f in val_neg_files:
  shutil.move(f, 'aclImdb/val/neg')

tokenizer = T5Tokenizer.from_pretrained('t5-base')
ids_neg = tokenizer.encode('negative </s>')
ids_pos = tokenizer.encode('positive </s>')
len(ids_neg), len(ids_pos)

# input: I went to see this movie with my husband, and we both thought the acting was terrible!"
# target: negative
# input: Despite what others say, I thought this movie was funny.
# target: positive


class ImdbDataset(Dataset):

  def __init__(self, tokenizer, data_dir, type_path, max_len=512):
    self.pos_file_path = os.path.join(data_dir, type_path, 'pos')
    self.neg_file_path = os.path.join(data_dir, type_path, 'neg')

    self.pos_files = glob.glob("%s/*.txt" % self.pos_file_path)
    self.neg_files = glob.glob("%s/*.txt" % self.neg_file_path)

    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask = self.inputs[index]["attention_mask"].squeeze(
    )  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze(
    )  # might need to squeeze

    return {
        "source_ids": source_ids,
        "source_mask": src_mask,
        "target_ids": target_ids,
        "target_mask": target_mask
    }

  def _build(self):
    self._buil_examples_from_files(self.pos_files, 'positive')
    self._buil_examples_from_files(self.neg_files, 'negative')

  def _buil_examples_from_files(self, files, sentiment):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for path in files:
      with open(path, 'r') as f:
        text = f.read()

      line = text.strip()
      line = REPLACE_NO_SPACE.sub("", line)
      line = REPLACE_WITH_SPACE.sub("", line)
      line = line + ' </s>'

      target = sentiment + " </s>"

      # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [line],
          max_length=self.max_len,
          pad_to_max_length=True,
          return_tensors="pt")
      # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=2, pad_to_max_length=True, return_tensors="pt")

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)


dataset = ImdbDataset(tokenizer, 'aclImdb', 'val', max_len=512)
len(dataset)

data = dataset[28]
print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))
# !mkdir -p t5_imdb_sentiment
args_dict.update({
    'data_dir': 'aclImdb',
    'output_dir': 't5_imdb_sentiment',
    'num_train_epochs': 2
})
args = argparse.Namespace(**args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.output_dir,
                                                   prefix="checkpoint",
                                                   monitor="val_loss",
                                                   mode="min",
                                                   save_top_k=5)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision=16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)


def get_dataset(tokenizer, type_path, args):
  return ImdbDataset(tokenizer=tokenizer,
                     data_dir=args.data_dir,
                     type_path=type_path,
                     max_len=args.max_seq_length)


model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)
# !mkdir t5_base_imdb_sentiment
model.model.save_pretrained('t5_base_imdb_sentiment')

# inference
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics
dataset = ImdbDataset(tokenizer, 'aclImdb', 'test', max_len=512)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
it = iter(loader)

batch = next(it)
batch["source_ids"].shape

outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                            attention_mask=batch['source_mask'].cuda(),
                            max_length=2)

dec = [tokenizer.decode(ids) for ids in outs]

texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

for i in range(32):
  lines = textwrap.wrap("Review:\n%s\n" % texts[i], width=100)
  print("\n".join(lines))
  print("\nActual sentiment: %s" % targets[i])
  print("Predicted sentiment: %s" % dec[i])
  print(
      "=====================================================================\n")

loader = DataLoader(dataset, batch_size=32, num_workers=4)
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
  outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                              attention_mask=batch['source_mask'].cuda(),
                              max_length=2)

  dec = [tokenizer.decode(ids) for ids in outs]
  target = [tokenizer.decode(ids) for ids in batch["target_ids"]]

  outputs.extend(dec)
  targets.extend(target)
# check whether the model generate any invalid text
for i, out in enumerate(outputs):
  if out not in ['positive', 'negative']:
    print(i, 'detected invalid prediction')

metrics.accuracy_score(targets, outputs)

# Emotion Classification
# targets: 'sadness', 'joy', 'anger', 'fear', 'surprise', 'love'.
# !wget https://www.dropbox.com/s/ikkqxfdbdec3fuj/test.txt
# !wget https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt
# !wget https://www.dropbox.com/s/2mzialpsgf9k5l3/val.txt
# !mkdir emotion_data
# !mv *.txt emotion_data
train_path = "emotion_data/train.txt"
test_path = "emotion_data/test.txt"
val_path = "emotion_data/val.txt"

## emotion labels
label2int = {
    "sadness": 0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5
}
data = pd.read_csv(train_path,
                   sep=";",
                   header=None,
                   names=['text', 'emotion'],
                   engine="python")
data.emotion.value_counts().plot.bar()
train.head()
train.count()

tokenizer = T5Tokenizer.from_pretrained('t5-base')

emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
for em in emotions:
  print(len(tokenizer.encode(em)))


class EmotionDataset(Dataset):

  def __init__(self, tokenizer, data_dir, type_path, max_len=512):
    self.path = os.path.join(data_dir, type_path + '.txt')

    self.data_column = "text"
    self.class_column = "emotion"
    self.data = pd.read_csv(self.path,
                            sep=";",
                            header=None,
                            names=[self.data_column, self.class_column],
                            engine="python")

    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask = self.inputs[index]["attention_mask"].squeeze(
    )  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze(
    )  # might need to squeeze

    return {
        "source_ids": source_ids,
        "source_mask": src_mask,
        "target_ids": target_ids,
        "target_mask": target_mask
    }

  def _build(self):
    for idx in range(len(self.data)):
      input_, target = self.data.loc[idx, self.data_column], self.data.loc[
          idx, self.class_column]

      input_ = input_ + ' </s>'
      target = target + " </s>"

      # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [input_],
          max_length=self.max_len,
          pad_to_max_length=True,
          return_tensors="pt")
      # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length=2, pad_to_max_length=True, return_tensors="pt")

      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)


dataset = EmotionDataset(tokenizer, 'emotion_data', 'val', 512)
len(dataset)

data = dataset[42]
print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))

!mkdir -p t5_emotion

args_dict.update({'data_dir': 'emotion_data', 'output_dir': 't5_emotion', 'num_train_epochs':2})
args = argparse.Namespace(**args_dict)
print(args_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

def get_dataset(tokenizer, type_path, args):
  return EmotionDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)


model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)

trainer.fit(model)
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

dataset = EmotionDataset(tokenizer, 'emotion_data', 'test', 512)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
it = iter(loader)
batch = next(it)
batch["source_ids"].shape

outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

dec = [tokenizer.decode(ids) for ids in outs]

texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

for i in range(32):
    c = texts[i]
    lines = textwrap.wrap("text:\n%s\n" % c, width=100)
    print("\n".join(lines))
    print("\nActual sentiment: %s" % targets[i])
    print("predicted sentiment: %s" % dec[i])
    print("=====================================================================\n")

dataset = EmotionDataset(tokenizer, 'emotion_data', 'test', 512)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
model.model.eval()
outputs = []
targets = []
for batch in tqdm(loader):
  outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

  dec = [tokenizer.decode(ids) for ids in outs]
  target = [tokenizer.decode(ids) for ids in batch["target_ids"]]
  
  outputs.extend(dec)
  targets.extend(target)

for i, out in enumerate(outputs):
  if out not in emotions:
    print(i, 'detected invalid prediction')

metrics.accuracy_score(targets, outputs)

print(metrics.classification_report(targets, outputs))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

cm = metrics.confusion_matrix(targets, outputs)

df_cm = pd.DataFrame(cm, index = ["anger", "fear", "joy", "love", "sadness", "surprise"], columns = ["anger", "fear", "joy", "love", "sadness", "surprise"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap='Purples', fmt='g')

