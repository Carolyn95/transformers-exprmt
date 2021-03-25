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
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import torch
# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)

import os
import shutil
import pdb

import datasets

from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)

import textwrap
from tqdm.auto import tqdm
from sklearn import metrics


# ========== SET SEED ==========
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


class MyDataset(Dataset):

  def __init__(self, tokenizer, data_dir, type_path, max_len=64):
    # type_path: train | test | val
    # each sentence is one file
    self.source_file_path = os.path.join(data_dir, type_path)  # , 'source'
    self.target_file_path = os.path.join(data_dir, type_path)  # , 'target'
    self.source_files = sorted(glob.glob("%s/*.source" % self.source_file_path))
    self.target_files = sorted(glob.glob("%s/*.target" % self.target_file_path))

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
    self._buil_examples_from_files(self.source_files, self.target_files)
    # self._buil_examples_from_files(self.neg_files, 'negative')

  def _buil_examples_from_files(self, source_files, target_files):
    # REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    # REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    # source_file = source_files[0]
    for source_file in source_files:
      with open(source_file, 'r') as f:
        text = f.read()
        line = text.strip()
        line = line + ' </s>'
      # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [line],
          max_length=self.max_len,
          pad_to_max_length=True,
          return_tensors="pt")
      self.inputs.append(tokenized_inputs)

    # target_file = target_files[0]
    for target_file in target_files:
      with open(target_file, 'r') as f:
        text = f.read()
        target = text.strip()
        target = target + ' </s>'
      # tokenize targets
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target],
          max_length=self.max_len,
          pad_to_max_length=True,
          return_tensors="pt")
      self.targets.append(tokenized_targets)


if __name__ == "__main__":
  args_dict = dict(
      data_dir="",  # path for data files
      output_dir="",  # path to save the checkpoints
      model_name_or_path='t5-base',  # t5-base | t5-small
      tokenizer_name_or_path='t5-base',  # t5-base | t5-small
      max_seq_length=64,
      learning_rate=3e-4,
      weight_decay=0.0,
      adam_epsilon=1e-8,
      warmup_steps=0,
      train_batch_size=8,  # 8 
      eval_batch_size=8,  # 8
      num_train_epochs=2,
      gradient_accumulation_steps=16,
      n_gpu=1,
      distributed_backend='ddp',  # not being used
      early_stop_callback=False,
      fp_16=
      False,  # if you want to enable 16-bit training then install apex and set this to true
      opt_level=
      'O1',  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
      max_grad_norm=
      1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
      seed=42,
  )

  tokenizer = T5Tokenizer.from_pretrained('t5-base')  # t5-base | t5-small

  # dataset = MyDataset(tokenizer,
  #                     './data/processed_full_conll_TokenEnt/',
  #                     'val',
  #                     max_len=64)

  # print('Length of dataset is {}'.format(len(dataset)))
  # data = dataset[0]
  # print(tokenizer.decode(data['source_ids']))
  # print(tokenizer.decode(data['target_ids']))

  paths = ['t5_conll_ner_new_envr',
           't5_base_conll_ner_new_envr']  # [out_path, model_path]

  # for path_ in paths:
  #   if not os.path.exists(path_):
  #     os.makedirs(path_)
  #   else:
  #     shutil.rmtree(path_)
  #     os.makedirs(path_)

  args_dict.update({
      'data_dir': './data/processed_full_conll_LabelOnly/',
      'output_dir': paths[1],
      'num_train_epochs': 10,
      'train_batch_size': 64,  # configurable
      'eval_batch_size': 64  # configurable
  })

  args = argparse.Namespace(**args_dict)

  checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.output_dir,
                                                     prefix="checkpoint",
                                                     monitor="val_loss",
                                                     mode="min",
                                                     save_top_k=5)

  train_params = dict(
      accumulate_grad_batches=args.gradient_accumulation_steps,
      gpus=args.n_gpu,  #distributed_backend=args.distributed_backend,
      max_epochs=args.num_train_epochs,
      early_stop_callback=False,
      precision=16 if args.fp_16 else 32,
      amp_level=args.opt_level,
      gradient_clip_val=args.max_grad_norm,
      checkpoint_callback=checkpoint_callback,
      callbacks=[LoggingCallback()],
  )

  def get_dataset(tokenizer, type_path, args):
    return MyDataset(tokenizer=tokenizer,
                     data_dir=args.data_dir,
                     type_path=type_path,
                     max_len=args.max_seq_length)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  model = T5ForConditionalGeneration.from_pretrained(paths[1]).to(device)

  # model = T5FineTuner(args)
  # trainer = pl.Trainer(**train_params)
  # trainer.fit(model)
  # model.model.save_pretrained('t5_base_10pctconll_ner')
  """
  # ==================== Eval on Train
  print('\n\n==================== Eval on Train ====================\n\n')

  # dataset = ImdbDataset(tokenizer, 'aclImdb', 'test',  max_len=512)
  dataset = MyDataset(tokenizer,
                      './data/processed_full_conll_LabelOnly/',
                      'train',
                      max_len=64)

  loader = DataLoader(dataset, batch_size=26, shuffle=True)
  it = iter(loader)
  batch = next(it)
  print(batch["source_ids"].shape)

  outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                              attention_mask=batch['source_mask'].cuda(),
                              max_length=64)

  dec = [tokenizer.decode(ids) for ids in outs]

  texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
  targets = [tokenizer.decode(ids) for ids in batch['target_ids']]
  for i in range(20):
    lines = textwrap.wrap("Source:\n%s\n" % texts[i], width=100)
    print("\n".join(lines))
    print("\nTarget : %s" % targets[i])
    print("\nPredicted : %s" % dec[i])
    print(
        "=====================================================================\n"
    )
  """
  # ==================== Eval on Eval

  print('\n\n==================== Eval on Eval ====================\n\n')
  # dataset = ImdbDataset(tokenizer, 'aclImdb', 'test',  max_len=512)
  dataset = MyDataset(
      tokenizer,
      './data/processed_full_conll_LabelOnly/',
      'val',  # test
      max_len=64)

  # loader = DataLoader(dataset, batch_size=26, shuffle=True)
  from torch.utils.data import SubsetRandomSampler
  splr = SubsetRandomSampler([i for i in range(len(dataset))])
  loader = DataLoader(dataset,
                      batch_size=16,
                      sampler=splr,
                      shuffle=False,
                      drop_last=True)
  metric = datasets.load_metric('seqeval')
  for i in range(len(loader)):
    it = iter(loader)

    batch = next(it)

    # batch["source_ids"].shape

    # outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
    #                             attention_mask=batch['source_mask'].cuda(),
    #                             max_length=64)
    outs = model.generate(input_ids=batch['source_ids'].cuda(),
                          attention_mask=batch['source_mask'].cuda(),
                          max_length=64)

    dec = [tokenizer.decode(ids).split() for ids in outs]
    # dec = [tokenizer.decode(ids) for ids in outs]

    # texts = [[tokenizer.decode(ids)] for ids in batch['source_ids']]
    # texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
    # targets = [tokenizer.decode(ids) for ids in batch['target_ids']]
    targets = [tokenizer.decode(ids).split() for ids in batch['target_ids']]
    # dec_flat = [d for dec_ in dec for d in dec_]
    # dec_uniq = list(set(dec_flat))
    uniq_labels = [
        'B-LOC', 'B-ORG', 'B-MISC', 'B-PER', 'I-LOC', 'I-ORG', 'I-MISC',
        'I-PER', 'O'
    ]
    # pdb.set_trace()
    for idx, (one_dec, one_target) in enumerate(zip(dec, targets)):
      one_dec = [_ for _ in one_dec if _ in uniq_labels]
      one_target = [_ for _ in one_target if _ in uniq_labels]
      # dec[idx] = one_dec_
      min_len = len(one_dec) if len(one_dec) <= len(one_target) else len(
          one_target)
      dec[idx] = one_dec[:min_len]
      targets[idx] = one_target[:min_len]

    metric.add_batch(predictions=dec, references=targets)
  score = metric.compute()
  print(score)

  # ==================== Eval on Test

  print('\n\n==================== Eval on Test ====================\n\n')
  # dataset = ImdbDataset(tokenizer, 'aclImdb', 'test',  max_len=512)
  dataset = MyDataset(
      tokenizer,
      './data/processed_full_conll_LabelOnly/',
      'test',  # test
      max_len=64)

  # loader = DataLoader(dataset, batch_size=26, shuffle=True)
  from torch.utils.data import SubsetRandomSampler
  splr = SubsetRandomSampler([i for i in range(len(dataset))])
  loader = DataLoader(dataset,
                      batch_size=16,
                      sampler=splr,
                      shuffle=False,
                      drop_last=True)
  metric = datasets.load_metric('seqeval')
  for i in range(len(loader)):
    it = iter(loader)

    batch = next(it)

    # batch["source_ids"].shape

    # outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
    #                             attention_mask=batch['source_mask'].cuda(),
    #                             max_length=64)
    outs = model.generate(input_ids=batch['source_ids'].cuda(),
                          attention_mask=batch['source_mask'].cuda(),
                          max_length=64)

    dec = [tokenizer.decode(ids).split() for ids in outs]
    # dec = [tokenizer.decode(ids) for ids in outs]

    # texts = [[tokenizer.decode(ids)] for ids in batch['source_ids']]
    # texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
    # targets = [tokenizer.decode(ids) for ids in batch['target_ids']]
    targets = [tokenizer.decode(ids).split() for ids in batch['target_ids']]
    # dec_flat = [d for dec_ in dec for d in dec_]
    # dec_uniq = list(set(dec_flat))
    uniq_labels = [
        'B-LOC', 'B-ORG', 'B-MISC', 'B-PER', 'I-LOC', 'I-ORG', 'I-MISC',
        'I-PER', 'O'
    ]
    # pdb.set_trace()
    for idx, (one_dec, one_target) in enumerate(zip(dec, targets)):
      one_dec = [_ for _ in one_dec if _ in uniq_labels]
      one_target = [_ for _ in one_target if _ in uniq_labels]
      # dec[idx] = one_dec_
      min_len = len(one_dec) if len(one_dec) <= len(one_target) else len(
          one_target)
      dec[idx] = one_dec[:min_len]
      targets[idx] = one_target[:min_len]

    metric.add_batch(predictions=dec, references=targets)
  score = metric.compute()
  print(score)

  # with open('LableOnly_test_all.pred', 'a') as f:
  #   for i in range(len(texts)):
  #     lines = textwrap.wrap("Source:\n%s\n" % texts[i], width=100)
  #     wrtable = ("\n".join(lines))
  #     wrtable += "\nTarget : %s" % targets[i]
  #     wrtable += "\nPredicted : %s\n" % dec[i]
  #     f.write(wrtable)
  #     f.write("\n")
  """
  from sklearn import metrics

  import datasets
  metric = datasets.load_metric('seqeval')

  for batch in dataset:
      inputs, references = batch
      predictions = model(inputs)
      metric.add_batch(predictions=predictions, references=references)
  score = metric.compute()

  """