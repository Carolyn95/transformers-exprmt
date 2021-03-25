"""
- NER(Named Entity Recognition): classify the entities in the text(person, organisation, location...)
- POS(Part-of-speech tagging): grammatically classify the tokens(noun, verb, adjective...)
- Chunk (Chunking): grammatically classify the tokens and group them into "chunks" that go together
20210203
gmb -> extra_ids (input same align as target)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import pdb
import datasets
from datasets import load_metric
import random
import pandas as pd
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
import pytorch_lightning as pl
import textwrap


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(2021)


class GmbData(Dataset):

  def __init__(self, tokenizer, data_dir, type_path, max_len=32):
    self.tokenizer = tokenizer
    self.data_dir = data_dir
    self.type_path = type_path
    self.text_file = data_dir + 'text_' + type_path + '.txt'
    self.label_file = data_dir + 'labels_' + type_path + '.txt'
    self.max_len = max_len
    self.org_inputs = []
    self.org_targets = []
    self.inputs = []  # aft tokenized
    self.targets = []  # aft tokenized
    self._build()

  def __getitem__(self, index):
    org_input = self.org_inputs[index]
    org_target = self.org_targets[index]
    tokenized_input = self.tokenizer.batch_encode_plus([org_input.strip()],
                                                       max_length=self.max_len,
                                                       pad_to_max_length=True,
                                                       return_tensors='pt')

    tokenized_target = self.tokenizer.batch_encode_plus([org_target.strip()],
                                                        max_length=self.max_len,
                                                        pad_to_max_length=True,
                                                        return_tensors='pt')

    self.inputs.append(tokenized_input)
    self.targets.append(tokenized_target)

    src_ids = tokenized_input["input_ids"].squeeze()
    src_msk = tokenized_input["attention_mask"].squeeze()
    tg_ids = tokenized_target["input_ids"].squeeze()
    tg_msk = tokenized_target["attention_mask"].squeeze()

    return {
        "source_ids": src_ids,
        "source_mask": src_msk,
        "target_ids": tg_ids,
        "target_mask": tg_msk,
    }

  def __len__(self):
    return len(self.org_inputs)

  def getorgitem(self, index):
    return {'sent': self.org_inputs[index], 'label': self.org_targets[index]}

  def _build(self):
    self._buil_examples_from_files(self.text_file, self.label_file)

  def _buil_examples_from_files(self, text_file, label_file):
    with open(text_file, 'r') as f:
      sents = f.read().strip().split('\n')

    with open(label_file, 'r') as f:
      labels = f.read().strip().split('\n')

    for one_sent, sent_label in zip(sents, labels):
      one_input = []
      one_target = []
      one_sent_tks = one_sent.split(' ')
      sent_label_tks = sent_label.split(' ')
      length = len(one_sent_tks)
      for sent_tk, label_tk, idx in zip(one_sent_tks, sent_label_tks,
                                        range(length)):
        xtra_tk = '<extra_id_' + str(idx) + '>'
        one_input.append(sent_tk.strip())
        one_input.append(xtra_tk)
        one_target.append(label_tk.strip())
        one_target.append(xtra_tk)
      org_input = ' '.join(one_input)
      org_target = ' '.join(one_target)
      self.org_inputs.append(org_input)
      self.org_targets.append(org_target)


class LitT5Finetuner(pl.LightningModule):

  def __init__(self, hparams, train_data, test_data):
    super(LitT5Finetuner, self).__init__()
    self.hparams = hparams
    self.train_data = train_data
    self.test_data = test_data
    # pdb.set_trace()
    self.model = T5ForConditionalGeneration.from_pretrained(
        hparams.model_name_or_path)
    self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

  def forward(self,
              input_ids,
              attention_mask=None,
              decoder_input_ids=None,
              decoder_attention_mask=None,
              labels=None):
    return self.model(
        input_ids,
        attention_mask=attention_mask,  
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels)

  def _step(self, batch):
    y = batch["target_ids"]
    # pdb.set_trace()
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
    train_loss = self._step(batch)
    # pdb.set_trace()
    self.log('train_loss',
             train_loss,
             on_epoch=True,
             prog_bar=True,
             logger=True)
    tensorboard_logs = {"train_loss": train_loss}
    return {
        "loss": train_loss,
        "log": tensorboard_logs
    }  # return value must contain key 'loss' # TODO: delete tensorboard_logs to see if the training continues

  def validation_step(self, batch, batch_idx):
    val_loss = self._step(batch)
    self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
    return {"val_loss": val_loss}

  def test_step(self, batch, batch_idx):
    test_loss = self._step(batch)
    self.log('test_loss', test_loss, on_epoch=True, prog_bar=True, logger=True)
    return {"test_loss": test_loss}

  def configure_optimizers(self):
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

  def train_dataloader(self):
    train_data_loader = DataLoader(self.train_data,
                                   batch_size=self.hparams.batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=1)
    return train_data_loader

  def val_dataloader(self):
    test_data_loader = DataLoader(self.test_data,
                                  batch_size=self.hparams.batch_size,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=1)
    return test_data_loader


if __name__ == "__main__":
  args_dict = dict(
      data_dir='data/gmb/',
      save_dir=None,
      model_name_or_path='t5-small',  # t5-base | t5-small
      tokenizer_name_or_path='t5-small',  # t5-base | t5-small
      max_seq_length=32,
      batch_size=16,
      learning_rate=3e-4,
      weight_decay=1e-2,
      adam_epsilon=1e-8,
      warmup_steps=0,
      train_batch_size=8,
      eval_batch_size=8,
      num_train_epochs=10,
      gradient_accumulation_steps=1,  # 16
      n_gpu=1,
      distributed_backend='ddp',  # not being used
      accelerator='ddp',
      early_stop_callback=False,
      opt_level='O1',
      max_grad_norm=1.0,
      seed=2021,
  )
  save_dir = args_dict['model_name_or_path'] + '_gmb_xtraid_new_envr_gpu_1/'
  args_dict.update({'save_dir': save_dir})
  args = argparse.Namespace(**args_dict)
  tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

  train_data = GmbData(tokenizer, args.data_dir, 'train', args.max_seq_length)
  test_data = GmbData(tokenizer, args.data_dir, 'dev', args.max_seq_length)
  subset_len = 300
  train_data.org_inputs = train_data.org_inputs[:subset_len]
  train_data.org_targets = train_data.org_targets[:subset_len]
  train_data.inputs = train_data.inputs[:subset_len]
  train_data.targets = train_data.targets[:subset_len]
  
  print(' ===> Length of train dataset is {}'.format(len(train_data)))
  print(' ===> Length of test dataset is {}'.format(len(test_data)))

  checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=args.save_dir,
                                                     prefix="checkpoint",
                                                     monitor="val_loss",
                                                     mode="min",
                                                     save_top_k=5)

  early_stop_callback = EarlyStopping(monitor='val_loss',
                                      min_delta=0.00,
                                      patience=3,
                                      verbose=False,
                                      mode='min')
  train_params = dict(
      accumulate_grad_batches=args.gradient_accumulation_steps,
      gpus=args.n_gpu,  #distributed_backend=args.distributed_backend,
      max_epochs=args.num_train_epochs,  # early_stop_callback=False,
      amp_level=args.opt_level,
      gradient_clip_val=args.max_grad_norm,
      checkpoint_callback=checkpoint_callback,
      callbacks=[early_stop_callback],
      enable_pl_optimizer=True)

  model = LitT5Finetuner(args, train_data, test_data)
  trainer = pl.Trainer(**train_params)
  # trainer.fit(model)

  # model.model.save_pretrained(args.save_dir)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = T5ForConditionalGeneration.from_pretrained(args.save_dir).to(device)

  print(
      '\n\n==================== Eval on Train One Batch ====================\n\n'
  )
  train_data_loader = DataLoader(train_data,
                                 batch_size=args.train_batch_size,
                                 shuffle=True)
  it = iter(train_data_loader)
  batch = next(it)
  outs = model.generate(input_ids=batch['source_ids'].cuda(),
                        attention_mask=batch['source_mask'].cuda())
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

  print(
      '\n\n==================== Eval on Test One Batch ====================\n\n'
  )
  test_data_loader = DataLoader(test_data,
                                batch_size=args.eval_batch_size,
                                shuffle=True)
  it = iter(test_data_loader)
  batch = next(it)
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

  print(
      '\n\n==================== Eval on Test Compute Matrics ====================\n\n'
  )
  from torch.utils.data import SubsetRandomSampler
  splr = SubsetRandomSampler([i for i in range(len(test_data))])
  loader = DataLoader(
      test_data,
      batch_size=args.eval_batch_size, 
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
    uniq_labels = ['B-GPE', 'B-TIME', 'B-PER', 'B-ORG', 'B-LOC', 'B-MISC', 'I-GPE', 'I-TIME', 'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'O']

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
