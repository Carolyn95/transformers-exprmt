"""
  TODO: implement same modeling architecture as PolyAI
  20210404
  banking 77
  PolyAI MLP design:
    one single 512-dimentional hidden layer
    train with SGD(stochastic gradient descent), 
    with the learning rate of 0.7 and linear decay
    rely on aggressive dropout 0.75 
    and train for 500 iterations to reach convergence

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import pdb
from torch.utils.data import Dataset, DataLoader
import datasets
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import random
import pandas as pd
import numpy as np
import torch
from torch import nn
import csv
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import transformers


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(2021)


class Banking77(datasets.GeneratorBasedBuilder):

  @staticmethod
  def load(data_dir=None, **kwargs):
    src_file = Path(__file__)
    if not data_dir:
      data_dir = src_file.with_suffix('')
    with open(Path(data_dir).joinpath('categories.json')) as fp:
      features = datasets.Features({
          'id': datasets.Value('string'),
          'text': datasets.Value('string'),
          'label': datasets.features.ClassLabel(names=json.load(fp))
      })
    return datasets.load_dataset(str(src_file.absolute()),
                                 data_dir=data_dir,
                                 features=features,
                                 **kwargs)

  def _info(self):
    return datasets.info.DatasetInfo()

  def _split_generators(self, dl_manager):
    data = dl_manager.download_and_extract({
        split: Path(self.config.data_dir).joinpath(f'{split}.csv')
        for split in ['train', 'test']
    })
    with open(data['train']) as fp:
      all_train = [x for x in csv.DictReader(fp)]
    with open(data['test']) as fp:
      test = [x for x in csv.DictReader(fp)]
    val_split = []
    labels = [x['category'] for x in all_train]
    for label, count in Counter(labels).items():
      n = count // 5
      val_split.extend([i for i, c in enumerate(labels) if c == label][:n])
    train_split = [i for i in range(len(all_train)) if i not in val_split]
    return [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={'samples': [all_train[i] for i in train_split]}),
        datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={'samples': [all_train[i] for i in val_split]}),
        datasets.SplitGenerator(name=datasets.Split.TEST,
                                gen_kwargs={'samples': test})
    ]

  def _generate_examples(self, samples):
    for i, eg in enumerate(samples):
      yield i, {'id': str(i), 'text': eg['text'], 'label': eg['category']}


class PolyAiMLP(nn.Module):

  def __init__(self, model_checkpoint, n_classes, hidden_layer_size):
    super(PolyAiMLP, self).__init__()
    self.hidden_layer_size = hidden_layer_size
    self.n_classes = n_classes
    self.bert = AutoModel.from_pretrained(model_checkpoint,
                                          output_hidden_states=True)
    self.hid = nn.Linear(
        self.bert.config.hidden_size, hidden_layer_size
    )  # 'dim' in distilbert config, 'hidden_size' in bert config
    self.relu_act = nn.ReLU()
    self.softmax_act = nn.Softmax(dim=1)
    self.drop = nn.Dropout(p=0.75)
    self.out = nn.Linear(hidden_layer_size, self.n_classes)

  def forward(self, input_ids, attention_mask, token_type_ids, labels):
    output = self.bert(input_ids=input_ids,
                       attention_mask=attention_mask)['pooler_output']
    output = self.drop(output)
    output = self.relu_act(self.hid(output))
    logits = self.softmax_act(self.out(output))
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, self.n_classes), labels.view(-1))

    return {"loss": loss, "logits": logits}


if __name__ == '__main__':
  # "bert-base-uncased" | "bert-base-cased"

  model_checkpoint = "bert-base-uncased"
  model_name = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

  def encode(data):
    return tokenizer(data['text'],
                     max_length=32,
                     padding='max_length',
                     truncation=True)

  tokenizer_batch_size = 16
  data_dir = 'data/banking77/banking_data'
  banking_data = Banking77.load(data_dir)

  train_datasets = banking_data['train']
  train_datasets = train_datasets.map(encode, batched=True)
  train_datasets = DataLoader(train_datasets,
                              batch_size=tokenizer_batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=2).dataset

  val_datasets = banking_data['validation']
  val_datasets = val_datasets.map(encode, batched=True)

  val_datasets = DataLoader(val_datasets,
                            batch_size=tokenizer_batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=2).dataset

  dev_datasets = banking_data['test']
  # dev_datasets = dev_datasets.map(encode, batched=True)

  label_list = banking_data['train'].features['label'].names
  hidden_layer_size = 512
  learning_rate = 0.7
  n_classes = len(label_list)
  model = PolyAiMLP(model_checkpoint,
                    n_classes,
                    hidden_layer_size=hidden_layer_size)

  banking_datasets = {}
  banking_datasets['train'] = train_datasets
  banking_datasets['val'] = val_datasets
  banking_datasets['test'] = dev_datasets

  is_grad = 'grad_'

  # for name, param in model.named_parameters():
  #   if 'classifier' not in name:  # classifier layer
  #     param.requires_grad = False


  def compute_metrics(pred):
    labels = pred.label_ids
    # output_hidden_states=False
    preds = pred.predictions.argmax(-1)
    # output_hidden_states=True
    # preds = [x.argmax(-1) for x in pred.predictions[0]]
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

  batch_size = 128
  exprmt_ds = 'banking77'
  traning_samples = len(train_datasets)
  output_dir = 'hf_' + is_grad + f'{model_name}' + f'_{exprmt_ds}' + f'_{traning_samples}'
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  args = TrainingArguments(
      output_dir=output_dir,
      evaluation_strategy='epoch',
      learning_rate=learning_rate,
      per_device_train_batch_size=batch_size,
      num_train_epochs=500,
      weight_decay=0.01,
      eval_accumulation_steps=16
  )  # model_parallel should use in conjunction with model.to('cuda')

  # ------------ Train with transformer trainer
  ## --------- train with customised optimizer and scheduler
  # mlp_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  # mlp_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
  #     optimizer=mlp_optimizer,
  #     num_warmup_steps=30,
  #     num_training_steps=args.num_train_epochs,
  #     power=1.0)
  # mlp_scheduler = transformers.get_linear_schedule_with_warmup(
  #     optimizer=mlp_optimizer,
  #     num_warmup_steps=50,
  #     num_training_steps=args.num_train_epochs)
  # optimizers = (mlp_optimizer, mlp_scheduler)
  # trainer = Trainer(model.to(device),
  #                   args,
  #                   train_dataset=banking_datasets['train'],
  #                   eval_dataset=banking_datasets['val'],
  #                   tokenizer=tokenizer,
  #                   compute_metrics=compute_metrics)  #,optimizers=optimizers
  # trainer.train()
  # trainer.save_model(args.output_dir)


  def train():
    print('training finished')

  def validate():
    print('validation finished')

  def save_trained_model():
    print('model saved')

  predictions = []
  labels = []

  banking_datasets['test'] = banking_datasets['test'].map(encode, batched=True)
  banking_datasets['test'].set_format(
      type='torch',
      columns=['input_ids', 'token_type_ids', 'label', 'attention_mask'])
  banking_datasets['test'] = DataLoader(banking_datasets['test'],
                                        batch_size=16,
                                        shuffle=False,
                                        drop_last=True,
                                        collate_fn=lambda x: x)

  with torch.no_grad():
    for ds_valid in banking_datasets['test']:
      batch_predictions, batch_labels, _ = trainer.predict(ds_valid)
      # output_hidden_states=True
      # batch_predictions = np.argmax(batch_predictions[0], axis=1)
      # output_hidden_states=False
      batch_predictions = batch_predictions.argmax(-1)
      batch_predictions = [p for p in batch_predictions]
      batch_labels = [b for b in batch_labels]
      predictions += batch_predictions
      labels += batch_labels

  results = accuracy_score(labels, predictions)
  print('ACCURACY RESULT: ', results)
