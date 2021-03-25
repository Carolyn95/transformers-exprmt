"""
- NER
- same training objective as ELECTRA
- uncased
- base for sequence-level task, full(default) for token-level task 0.013969998854918125
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pdb
from datasets import load_dataset, load_metric
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
import torch

task = 'ner'  # ['ner', 'pos', 'chunk']
# "funnel-transformer/small-base", 
# "funnel-transformer/intermediate", "funnel-transformer/xlarge-base"
# "funnel-transformer/medium-base", "funnel-transformer/large-base"

model_checkpoint = "funnel-transformer/xlarge"
model_name = "funnel-xlarge"
datasets = load_dataset('conll2003')

label_list = datasets['train'].features[f"{task}_tags"].feature.names
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
label_all_tokens = True
# 75, 150, 300, 1500, len(datasets['train'])
traning_samples = 1500
temp = datasets['train'].select([i for i in range(traning_samples)])
datasets['train'] = temp


def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples["tokens"],
                               truncation=True,
                               is_split_into_words=True)

  labels = []
  for i, label in enumerate(examples[f"{task}_tags"]):
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
      # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
      if word_idx is None:
        label_ids.append(-100)
      # We set the label for the first token of each word.
      elif word_idx != previous_word_idx:
        label_ids.append(label[word_idx])
      # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
      else:
        label_ids.append(label[word_idx] if label_all_tokens else -100)
      previous_word_idx = word_idx

    labels.append(label_ids)

  tokenized_inputs["labels"] = labels
  return tokenized_inputs


# tokenize_and_align_labels(datasets['train'][:5])
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list))

# no_grad
is_grad = 'nograd_'
for name, param in model.named_parameters():
  if 'classifier' not in name:  # classifier layer
    param.requires_grad = False


data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric('seqeval')
# labels = [label_list[i] for i in example[f'{task}_tags']]
# metric.compute(predictions=[labels], references=[labels])


def compute_metrics(p):
  predictions, labels = p
  predictions = np.argmax(predictions, axis=2)
  true_predictions = [[
      label_list[p] for (p, l) in zip(prediction, label) if l != -100
  ] for prediction, label in zip(predictions, labels)]
  true_labels = [[
      label_list[l] for (p, l) in zip(prediction, label) if l != -100
  ] for prediction, label in zip(predictions, labels)]
  results = metric.compute(predictions=true_predictions, references=true_labels)
  return {
      'precision': results['overall_precision'],
      'recall': results['overall_recall'],
      'f1': results['overall_f1'],
      'accuracy': results['overall_accuracy']
  }


batch_size = 8
exprmt_ds = 'conll'
output_dir = 'hf_'  + is_grad + f'{model_name}' + f'_{exprmt_ds}' + f'_{task}' + f'_{traning_samples}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    model_parallel=True
)  # model_parallel should use in conjunction with model.to('cuda')
trainer = Trainer(model.to(device),
                  args,
                  train_dataset=tokenized_datasets['train'],
                  eval_dataset=tokenized_datasets['validation'],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)
trainer.train()
trainer.save_model(args.output_dir)
eval_result = trainer.evaluate()
print(eval_result)

predictions, labels, _ = trainer.predict(tokenized_datasets['test'])
predictions = np.argmax(predictions, axis=2)
true_predictions = [[
    label_list[p] for (p, l) in zip(prediction, label) if l != -100
] for prediction, label in zip(predictions, labels)]
true_labels = [[
    label_list[l] for (p, l) in zip(prediction, label) if l != -100
] for prediction, label in zip(predictions, labels)]
results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)
