"""
- NER(Named Entity Recognition): classify the entities in the text(person, organisation, location...)
- POS(Part-of-speech tagging): grammatically classify the tokens(noun, verb, adjective...)
- Chunk (Chunking): grammatically classify the tokens and group them into "chunks" that go together
"""

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

task = 'ner'  # ['ner', 'pos', 'chunk']

model_checkpoint = "base-cased"
batch_size = 16
datasets = load_dataset('conll2003')
label_list = datasets['train'].features[f"{task}_tags"].feature.names
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
example = datasets['train'][4]
tokenized_input = tokenizer(example['tokens'], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
word_ids = tokenized_input.word_ids()
aligned_labels = [
    -100 if i is None else example[f'{task}_tags'][i] for i in word_ids
]
print(len(aligned_labels), len(tokenized_input['input_ids']))
label_all_tokens = True


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


tokenize_and_align_labels(datasets['train'][:5])
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list))

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = load_metric('seqeval')
labels = [label_list[i] for i in example[f'{task}_tags']]
metric.compute(predictions=[labels], references=[labels])


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


args = TrainingArguments(f'test-{task}',
                         evaluation_strategy='epoch',
                         learning_rate=2e-5,
                         per_device_train_batch_size=batch_size,
                         per_device_eval_batch_size=batch_size,
                         num_train_epochs=3,
                         weight_decay=0.01)
trainer = Trainer(model,
                  args,
                  train_dataset=tokenized_datasets['train'],
                  eval_dataset=tokenized_datasets['validation'],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)
trainer.train()
trainer.evaluate()

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
