"""Tokenize Dataset or DatasetDict with 'text' index containing string and 'label' index containing integers
"""
import os
from datasets.dataset_dict import DatasetDict


def tokenize_data(tokenizer, data, lowercase=False):
  """Add 'input_ids' field to dataset. 

  Args:
     tokenizer: instance of tokenizer
     data: instance of Dataset or DatasetDict, containing text('String') and label('integer')
     lowercase: whether to lowercase text or not before tokenization
  
  Returns:
     Same container as data with additional fields 'input_ids' and optinally, 
     'token_type_ids', 'attention_mask'
  """

  def tokenize(examples):
    if lowercase:
      text = [eg.lower() for eg in examples['text']]
    else:
      text = examples['text']
    return tokenizer(text)

  # Number of processes is half of CPUs or size of smallest subsets
  if isinstance(data, DatasetDict):
    min_samples = min([len(split) for split in data])
  else:
    min_samples = len(data)
  num_proc = min(os.cpu_count() // 2, min_samples)
  return data.map(tokenize, batched=True, num_proc=num_proc)
