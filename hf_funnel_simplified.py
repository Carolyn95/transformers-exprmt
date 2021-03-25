import torch
# import tensorflow as tf
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import pdb
import numpy as np
import pickle as pkl
# from tensorflow.keras.layers import Input, Lambda, Dense, Dropout
import time
data_dir = 'data/atfm-data'
with open(data_dir + '/n_labels.pkl', 'rb') as f:
  n_labels = pkl.load(f)

train_sents = np.load(data_dir + '/train_sents.npy', allow_pickle=True)
# pdb.set_trace()
train_labels = np.load(data_dir + '/train_intents.npy', allow_pickle=True)

test_sents = np.load(data_dir + '/eval_sents.npy', allow_pickle=True)
test_labels = np.load(data_dir + '/eval_intents.npy', allow_pickle=True)

model_checkpoint = "bert-base-uncased"
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                           num_labels=n_labels)

train_encodings = tokenizer(train_sents, truncation=True, padding=True)
test_encodings = tokenizer(test_sents, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(train_encodings), train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(test_encodings), test_labels))

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    model_parallel=True
)  # model_parallel should use in conjunction with model.to('cuda')

from sklearn.metrics import accuracy_score


def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


trainer = Trainer(model.to(device),
                  args,
                  train_dataset=train_sents,
                  eval_dataset=test_sents,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(args.output_dir)
