# transformers==4.1.1 & pytorch-lightning==1.1.3
# ref from pytorch-lightening: https://github.com/huggingface/transformers/blob/master/examples/lightning_base.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import re
import gc
import pdb
import json
import time
import glob
import nltk
# nltk.download('punkt')
import torch
import random
import shutil
import logging
import textwrap
import argparse
import datasets
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import chain
from sklearn import metrics
import pytorch_lightning as pl
from string import punctuation
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(42)
logger = logging.getLogger(__name__)


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
  # Remove columns that are populated exclusively by pad_token_id
  keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
  if attention_mask is None:
    return input_ids[:, keep_column_mask]
  else:
    return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class T5FineTuner(pl.LightningModule):

  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.hparams = hparams

    self.model = T5ForConditionalGeneration.from_pretrained(
        hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

  def is_logger(self):
    return self.trainer.global_rank <= 0

  def forward(self,
              input_ids,
              attention_mask=None,
              decoder_input_ids=None,
              decoder_attention_mask=None,
              labels=None):
    return self.model(
        input_ids,
        attention_mask=attention_mask,  # decoder_input_ids=decoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
    )

  def _step(self, batch):
    y = batch["target_ids"]
    lm_labels = y[:, :].clone()
    lm_labels[y[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        labels=lm_labels,  # decoder_input_ids=y_ids,
        decoder_attention_mask=batch['target_mask'])

    loss = outputs['loss']
    return loss

  def training_step(self, batch, batch_idx):
    # after training `gc.collect(); torch.cuda.empty_cache();`
    loss = self._step(batch)
    self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
    self.trainer.train_loop.running_loss.append(loss)
    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}

  def validation_step(self, batch, batch_idx):
    val_loss = self._step(batch)
    self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
    return {"val_loss": val_loss}

  def configure_optimizers(self):
    # Prepare optimizer and schedule (linear warmup and decay)
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
                     optimizer_closure,
                     on_tpu=None,
                     using_native_amp=None,
                     using_lbfgs=None):
    optimizer.step(closure=optimizer_closure)
    optimizer.zero_grad()
    self.lr_scheduler.step()

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer,
                                type_path="train",
                                args=self.hparams)
    dataloader = DataLoader(train_dataset,
                            batch_size=self.hparams.train_batch_size,
                            drop_last=True,
                            shuffle=True,
                            num_workers=1)
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
                      drop_last=True,
                      shuffle=True,
                      num_workers=1)


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
      output_test_results_file = os.path.join(pl_module.hparams.output_dir,
                                              "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))


class MyDataset(Dataset):

  def __init__(self, tokenizer, data_dir, type_path, max_len=32):
    # type_path: train | test | val; each sentence is one file
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
    src_mask = self.inputs[index]["attention_mask"].squeeze()
    target_mask = self.targets[index]["attention_mask"].squeeze()

    return {
        "source_ids": source_ids,
        "source_mask": src_mask,
        "target_ids": target_ids,
        "target_mask": target_mask
    }

  def _build(self):
    self._buil_examples_from_files(self.source_files, self.target_files)

  def _buil_examples_from_files(self, source_files, target_files):

    for source_file in source_files:
      with open(source_file, 'r') as f:
        text = f.read()
        line = text.strip()
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [line],
            max_length=self.max_len,
            pad_to_max_length=True,
            return_tensors="pt")
        self.inputs.append(tokenized_inputs)

    for target_file in target_files:
      with open(target_file, 'r') as f:
        text = f.read()
        target = text.strip()
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target],
            max_length=self.max_len,
            pad_to_max_length=True,
            return_tensors="pt")
        self.targets.append(tokenized_targets)

  @staticmethod
  def trim_seq2seq_batch(batch, pad_token_id, test=False):
    # Remove columns that are populated exclusively by pad_token_id
    # This ensures that each batch is padded only uptil the "max sequence length"
    # https://github.com/huggingface/transformers/blob/1e51bb717c04ca4b01a05a7a548e6b550be38628/src/transformers/tokenization_utils.py
    source_ids, source_mask = trim_batch(batch["source_ids"],
                                         pad_token_id,
                                         attention_mask=batch["source_mask"])
    if test:
      return source_ids, source_mask, None
    y = trim_batch(batch["target_ids"], pad_token_id)
    return source_ids, source_mask, y


def get_dataset(tokenizer, type_path, args):
  return MyDataset(tokenizer=tokenizer,
                   data_dir=args.data_dir,
                   type_path=type_path,
                   max_len=args.max_seq_length)


if __name__ == "__main__":
  # -------------------------- paths & params & callbacks -------------------------- #
  paths = [
      't5_conll_300_xtraid_new_envr_gpu_1',
      't5_base_conll_300_xtraid_new_envr_gpu_1'
  ]  # [out_path, model_path]

  for path_ in paths:
    if not os.path.exists(path_):
      os.makedirs(path_)
    else:
      shutil.rmtree(path_)
      os.makedirs(path_)

  args_dict = dict(
      data_dir="./data/processed_300_conll_XtraID/",  # path for data files
      output_dir=paths[1],  # path to save the checkpoints
      model_name_or_path='t5-small',  # t5-base | t5-small
      tokenizer_name_or_path='t5-small',  # t5-base | t5-small
      max_seq_length=32,  # 32
      learning_rate=3e-4,
      weight_decay=1e-2,  # 1e-2 | 0.0
      adam_epsilon=1e-8,
      warmup_steps=0,
      train_batch_size=16,  # 8 
      eval_batch_size=16,  # 8
      num_train_epochs=10,
      gradient_accumulation_steps=1,  # 16
      n_gpu=1,
      distributed_backend='ddp',  # not being used
      accelerator='ddp',
      early_stop_callback=False,
      opt_level=
      'O1',  # get more optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
      max_grad_norm=1.0,
      seed=42,
  )
  # if required to update arguments
  # args_dict.update({
  #     'data_dir': './data/processed_full_conll_LabelOnly/',
  #     'output_dir': 't5_10pctconll_ner_new_envr',
  #     'num_train_epochs': 3,
  #     'n_gpu': 1,
  #     'train_batch_size': 64,  # configurable
  #     'eval_batch_size': 64  # configurable
  # })

  args = argparse.Namespace(**args_dict)

  checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.output_dir,
                                                     prefix="checkpoint",
                                                     monitor="val_loss",
                                                     mode="min",
                                                     save_top_k=5)

  early_stop_callback = EarlyStopping(monitor='val_loss',
                                      min_delta=0.00,
                                      patience=3,
                                      verbose=False,
                                      mode='min')

  # -------------------------- sanity check conll -------------------------- #
  tokenizer = T5Tokenizer.from_pretrained(
      args.tokenizer_name_or_path)  # t5-base | t5-small

  dataset = MyDataset(tokenizer,
                      args.data_dir,
                      'val',
                      max_len=args.max_seq_length)

  print('Length of dataset is {}'.format(len(dataset)))
  data = dataset[0]
  print(tokenizer.decode(data['source_ids'], skip_special_tokens=True))
  print(tokenizer.decode(data['target_ids'], skip_special_tokens=True))

  # -------------------------- start -------------------------- #
  train_params = dict(
      accumulate_grad_batches=args.gradient_accumulation_steps,
      gpus=args.n_gpu,  #distributed_backend=args.distributed_backend,
      max_epochs=args.num_train_epochs,  # early_stop_callback=False,
      amp_level=args.opt_level,
      gradient_clip_val=args.max_grad_norm,
      checkpoint_callback=checkpoint_callback,
      callbacks=[early_stop_callback, LoggingCallback()],
      enable_pl_optimizer=True)

  model = T5FineTuner(args)
  trainer = pl.Trainer(**train_params)
  trainer.fit(model)

  model.model.save_pretrained(paths[1])
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  # model.to(device)
  model = T5ForConditionalGeneration.from_pretrained(paths[1]).to(device)
  # torch.cuda.empty_cache()
  # torch.cuda.memory_summary(device=None, abbreviated=False)

  # -------------------------- Eval on Train -------------------------- #
  print('\n\n==================== Eval on Train ====================\n\n')

  dataset = MyDataset(tokenizer, args.data_dir, 'train')  # ,max_len=64

  loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
  it = iter(loader)
  batch = next(it)
  # 1 =====
  # outs = model.model.generate(
  #     input_ids=batch['source_ids'].cuda(),
  #     attention_mask=batch['source_mask'].cuda())  # max_length=64
  # 2 =====
  outs = model.generate(
      input_ids=batch['source_ids'].cuda(),
      attention_mask=batch['source_mask'].cuda())  # max_length=64
  # 3 =====
  # source_ids, source_mask, _ = MyDataset.trim_seq2seq_batch(
  #     batch, tokenizer.pad_token_id, test=True)
  # outs = model.generate(
  #     input_ids=source_ids,
  #     attention_mask=source_mask,
  #     num_beams=1,
  #     max_length=80,
  #     repetition_penalty=2.5,
  #     length_penalty=1.0,
  #     early_stopping=True,
  #     use_cache=True,
  # )

  dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
  # dec = tokenizer.batch_decode(outs, skip_special_tokens=True)

  texts = [
      tokenizer.decode(ids, skip_special_tokens=True)
      for ids in batch['source_ids']
  ]
  targets = [
      tokenizer.decode(ids, skip_special_tokens=True)
      for ids in batch['target_ids']
  ]
  for i in range(args.train_batch_size):
    lines = textwrap.wrap("Source:\n%s\n" % texts[i], width=100)
    print("\n".join(lines))
    print("\nTarget : %s" % targets[i])
    print("\nPredicted : %s" % dec[i])
    print(
        "=====================================================================\n"
    )

  # -------------------------- Eval on Val -------------------------- #
  print('\n\n==================== Eval on Val ====================\n\n')

  dataset = MyDataset(tokenizer,
                      args.data_dir,
                      'val',
                      max_len=args.max_seq_length)  #

  loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True)
  it = iter(loader)
  batch = next(it)
  batch["source_ids"].shape

  # outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
  #                             attention_mask=batch['source_mask'].cuda(),
  #                             max_length=16)
  outs = model.generate(input_ids=batch['source_ids'].cuda(),
                        attention_mask=batch['source_mask'].cuda(),
                        max_length=args.max_seq_length)

  dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

  texts = [
      tokenizer.decode(ids, skip_special_tokens=True)
      for ids in batch['source_ids']
  ]
  targets = [
      tokenizer.decode(ids, skip_special_tokens=True)
      for ids in batch['target_ids']
  ]
  for i in range(args.train_batch_size):
    lines = textwrap.wrap("Source:\n%s\n" % texts[i], width=100)
    print("\n".join(lines))
    print("\nTarget : %s" % targets[i])
    print("\nPredicted : %s" % dec[i])
    print(
        "=====================================================================\n"
    )

# -------------------------- Eval on Test -------------------------- #
  print('\n\n==================== Eval on Test ====================\n\n')
  dataset = MyDataset(tokenizer,
                      args.data_dir,
                      'test',
                      max_len=args.max_seq_length)

  # loader = DataLoader(dataset, batch_size=26, shuffle=True)
  from torch.utils.data import SubsetRandomSampler
  splr = SubsetRandomSampler([i for i in range(len(dataset))])
  loader = DataLoader(
      dataset,
      batch_size=args.eval_batch_size,  # 16,
      sampler=splr,
      shuffle=False,  # sampler is mutually exclusive with shuffle
      drop_last=True)
  metric = datasets.load_metric('seqeval')
  for i in range(len(loader)):
    it = iter(loader)

    batch = next(it)
    outs = model.generate(input_ids=batch['source_ids'].cuda(),
                          attention_mask=batch['source_mask'].cuda(),
                          max_length=args.max_seq_length)

    dec = [
        tokenizer.decode(ids, skip_special_tokens=True).split() for ids in outs
    ]

    targets = [
        tokenizer.decode(ids, skip_special_tokens=True).split()
        for ids in batch['target_ids']
    ]
    # dec_flat = [d for dec_ in dec for d in dec_]
    # dec_uniq = list(set(dec_flat))
    uniq_labels = [
        'B-LOC', 'B-ORG', 'B-MISC', 'B-PER', 'I-LOC', 'I-ORG', 'I-MISC',
        'I-PER', 'O'
    ]

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
