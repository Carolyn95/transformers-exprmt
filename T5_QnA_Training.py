# Sentiment span extraction using T5
# !pip install transformers==2.9.1
# !pip install pytorch-lightning==0.7.6
# upload data `sample_submission.csv | test.csv | train.csv ` and store in folder `tweet-sentiment-extraction/`

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
# Row 314 of train set is nan
train = pd.read_csv('/content/tweet-sentiment-extraction/train.csv').dropna()  #.head(1000)
test = pd.read_csv('/content/tweet-sentiment-extraction/test.csv')  #.head(1000)

# Set random 13% as the validation set (make validation set similar in size to test set)
train, val = train_test_split(train, test_size=0.13, random_state=42)

train.shape, test.shape, val.shape

# Confirm that the train set doesn't have any overlaps with the test and validation sets
set(test.textID.values).intersection(train.textID.values), set(
    val.textID.values).intersection(train.textID.values)
train.head(), test.head(), val.head()

train.columns

# Input
for a, b, _ in zip(train.sentiment.values[:10], train.text.values[:10],
                   train.selected_text.values[:10]):
  print("sentiment:", a, "tweet:", b)

# Target (what we're trying to predict)
for _, _, c in zip(train.sentiment.values[:10], train.text.values[:10],
                   train.selected_text.values[:10]):
  print(c)

# Checking for NaNs
train.isna().sum().sum(), test.isna().sum().sum(), val.isna().sum().sum()

# Append EOS token to target text
# This is the standard format for T5 targets
# More info in transformers docs: https://huggingface.co/transformers/model_doc/t5.html
train['selected_text'] = train['selected_text'] + ' </s>'
val['selected_text'] = val['selected_text'] + ' </s>'

# Apply Q&A structure
# From Appendix D in the T5 paper
processed_input_train = ("question: " + train.sentiment + " context: " +
                         train.text)
processed_input_test = ("question: " + test.sentiment + " context: " +
                        test.text)
processed_input_val = ("question: " + val.sentiment + " context: " + val.text)

# Save data as string separated by \n (new line)
processed_input_str_train = '\n'.join(processed_input_train.values.tolist())
processed_input_str_test = '\n'.join(processed_input_test.values.tolist())
selected_text_str_train = '\n'.join(train['selected_text'].values.tolist())
processed_input_str_val = '\n'.join(processed_input_val.values.tolist())
selected_text_str_val = '\n'.join(val['selected_text'].values.tolist())

processed_input_train[0], train['selected_text'][0]

processed_input_test[0]

# Save source files

with open('/content/train.source', 'w') as f:
  f.write(processed_input_str_train)

# Making dev similar in this case
with open('/content/test.source', 'w') as f:
  f.write(processed_input_str_test)

with open('/content/val.source', 'w') as f:
  f.write(processed_input_str_val)

# !head /content/train.source | !head /content/test.source | !head /content/val.source

with open('/content/train.target', 'w') as f:
  f.write(selected_text_str_train)

with open('/content/val.target', 'w') as f:
  f.write(selected_text_str_val)

# !head /content/train.target | !head /content/test.target | !head /content/val.target

# Preparing T5 dataset (applicable for both T5 & BART)
import os

import torch
from torch.utils.data import Dataset

from transformers.tokenization_utils import trim_batch


def encode_file(tokenizer,
                data_path,
                max_length,
                pad_to_max_length=True,
                return_tensors="pt"):
  """
  This function reads the text files that we prepared and returns them in tokenized form.

  Actually tokenizer.batch_encode_plus returns these as a list of dictionaries where 
  each dictionary contains the word piece indices among other relevant inputs for training & inference
  """
  examples = []
  with open(data_path, "r") as f:
    for text in f.readlines():
      tokenized = tokenizer.batch_encode_plus(
          [text],
          max_length=max_length,
          pad_to_max_length=pad_to_max_length,
          return_tensors=return_tensors,
      )
      examples.append(tokenized)
  return examples


class T5Dataset(Dataset):
  """
  This is the T5 dataset that can read our train, test, and dev files separately

  This was patterned after the SummarizationDataset from the `transformer` library's summarization example (compatible with T5)
  """

  def __init__(self,
               tokenizer,
               data_dir="../working/",
               type_path="train",
               max_source_length=1024,
               max_target_length=56):
    super().__init__()
    # Store the tokenizer
    self.tokenizer = tokenizer
    self.type_path = type_path
    # Read the source and target files for the type of file (train, test, or val)
    self.source = encode_file(tokenizer,
                              os.path.join(data_dir, type_path + ".source"),
                              max_source_length)
    self.target = None
    if self.type_path != "test":
      self.target = encode_file(tokenizer,
                                os.path.join(data_dir, type_path + ".target"),
                                max_target_length)

  def __len__(self):
    return len(self.source)

  def __getitem__(self, index):
    # Return example as a dictionary containing source_ids, src_mask, and target_ids
    source_ids = self.source[index]["input_ids"].squeeze()  # (1024,)
    # We need masks for transformers to:
    # 1) ignore padding for both the encoder and decoder stages (src_mask)
    # 2) ignore future tokens at the decoder stage
    src_mask = self.source[index]["attention_mask"].squeeze()

    if self.type_path == "test":
      return {"source_ids": source_ids, "source_mask": src_mask}

    target_ids = self.target[index]["input_ids"].squeeze()  # (56, )
    return {
        "source_ids": source_ids,
        "source_mask": src_mask,
        "target_ids": target_ids
    }

  # Static methods, much like class methods, are methods that are bound to a class rather than its object.
  # They do not require a class instance creation. So, they are not dependent on the state of the object.
  # https://www.programiz.com/python-programming/methods/built-in/staticmethod
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

  def collate_fn(self, batch):
    """
    The tensors are stacked together as they are yielded.

    Collate function is applied to the output of a DataLoader as it is yielded.
    """
    input_ids = torch.stack([x["source_ids"] for x in batch])  # BS x SL
    masks = torch.stack([x["source_mask"] for x in batch])  # BS x SL
    pad_token_id = self.tokenizer.pad_token_id
    source_ids, source_mask = trim_batch(input_ids,
                                         pad_token_id,
                                         attention_mask=masks)
    if self.type_path == "test":
      return {"source_ids": source_ids, "source_mask": source_mask}

    target_ids = torch.stack([x["target_ids"] for x in batch])  # BS x SL
    # Remove columns that are purely padding
    y = trim_batch(target_ids, pad_token_id)
    # Return dictionary containing tensors
    return {
        "source_ids": source_ids,
        "source_mask": source_mask,
        "target_ids": y
    }


# Model
import argparse
import logging
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def set_seed(args: argparse.Namespace):
  """
    Set all the seeds to make results replicable
    """
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def jaccard(str1, str2):
  a = set(str1.lower().split())
  b = set(str2.lower().split())
  c = a.intersection(b)
  return float(len(c)) / (len(a) + len(b) - len(c))


class T5Module(pl.LightningModule):
  """
    Base Transformer model that uses Pytorch Lightning as a PyTorch wrapper.

    T5 specific methods are implemented in T5Trainer
    """

  def __init__(self, hparams: argparse.Namespace, **config_kwargs):
    "Initialize a model."

    super().__init__()
    self.hparams = hparams
    cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
    # Read the config file of the T5 model (T5Config)
    # AutoConfig allows you to read the configuration for a specified model (e.g. in this case, t5-base)
    # Reference: https://huggingface.co/transformers/model_doc/auto.html#autoconfig
    self.config = AutoConfig.from_pretrained(self.hparams.model_name_or_path)
    # Read the tokenizer of the T5 model (T5Tokenizer)
    # AutoTokenizer allows you to read the tokenizer for a specified model (e.g. in this case, t5-base)
    # Reference: https://huggingface.co/transformers/model_doc/t5.html#t5tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.hparams.model_name_or_path,
        cache_dir=cache_dir,
    )
    # Read the model file for the pre-trained T5 model (T5ForConditionalGeneration)
    # AutoModelWithLMHead allows you to read any of the language modelling models from the transformers library (e.g. in this case, t5-base)
    # Automodels reference: https://huggingface.co/transformers/model_doc/auto.html#automodel
    self.model = AutoModelWithLMHead.from_pretrained(
        self.hparams.model_name_or_path,
        from_tf=bool(".ckpt" in self.hparams.model_name_or_path
                    ),  # Checkpoint is a TF format
        config=self.config,
        cache_dir=cache_dir,
    )

    # Save dataset params
    self.dataset_kwargs: dict = dict(
        data_dir=self.hparams.data_dir,
        max_source_length=self.hparams.max_source_length,
        max_target_length=self.hparams.max_target_length,
    )

  # Forward function
  # Defines the forward pass of the module

  def forward(
      self,
      input_ids,  # Indices of input sequence tokens in the vocabulary. 
      attention_mask=None,  # Mask to avoid performing attention on padding token indices
      decoder_input_ids=None,  # T5 uses the pad_token_id as the starting token for decoder_input_ids generation.
      lm_labels=None  # Labels for computing the sequence classification/regression loss (see T5Model). Note: loss is returned when lm_label is provided.
  ):
    """
         loss (torch.FloatTensor of shape (1,), optional, returned when lm_label is provided
        """
    # Details on how to use this in the Hugging Face T5 docs: https://huggingface.co/transformers/model_doc/t5.html
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        lm_labels=lm_labels,
    )

  # Data preparation

  def get_dataloader(self,
                     type_path: str,
                     batch_size: int,
                     shuffle: bool = False) -> DataLoader:
    dataset = T5Dataset(self.tokenizer,
                        type_path=type_path,
                        **self.dataset_kwargs)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=dataset.collate_fn,
                            shuffle=shuffle)
    return dataloader

  def train_dataloader(self) -> DataLoader:
    dataloader = self.get_dataloader("train",
                                     batch_size=self.hparams.train_batch_size,
                                     shuffle=True)
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

  def val_dataloader(self) -> DataLoader:
    return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

  def test_dataloader(self) -> DataLoader:
    return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

  # Configure optimizers

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    # Weight decay explanation:
    # Weight decay will not be applied to "bias" and "LayerNorm.weight" parameters
    # When training neural networks, it is common to use "weight decay," where after each update,
    # the weights are multiplied by a factor slightly less than 1.
    # This prevents the weights from growing too large, and can be seen as gradient descent on a quadratic regularization term.
    # https://metacademy.org/graphs/concepts/weight_decay_neural_networks
    no_decay = ["bias", "LayerNorm.weight"]

    # Group parameters to those that will and will not have weight decay applied
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
    # Use AdamW as an optimizer
    # Intro here: https://www.fast.ai/2018/07/02/adam-weight-decay/
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=self.hparams.learning_rate,
                      eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]

  # Forward pass and calculate loss per batch (step)

  def _step(self, batch, return_text=False):
    """
        Runs forward pass and calculates loss per batch. Applied for training_step, and validation_step
    """
    pad_token_id = self.tokenizer.pad_token_id
    source_ids, source_mask, y = batch["source_ids"], batch[
        "source_mask"], batch["target_ids"]
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone()
    # Change pad_token_id to -100
    lm_labels[y[:, 1:] == pad_token_id] = -100
    # Run forward pass and calculate loss
    outputs = self(
        source_ids,
        attention_mask=source_mask,
        decoder_input_ids=y_ids,
        lm_labels=lm_labels,
    )
    # Only get loss from the output since that's all we need to apply our optimizer
    loss = outputs[0]
    if return_text:
      target_text = [self.tokenizer.decode(ids) for ids in y_ids]
      return loss, target_text
    else:
      return loss

  # Step during training

  def training_step(self, batch, batch_idx):
    """
        Runs forward pass, calculates loss, and returns loss (and logs) in a dict
        """
    loss = self._step(batch)

    # Notice that each training step loss is recorded on tensorboard, which makes sense since we're tracking loss per batch
    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}

  # Adjust weights based on calculated gradients and learning rate scheduler

  def optimizer_step(self,
                     epoch,
                     batch_idx,
                     optimizer,
                     optimizer_idx,
                     second_order_closure=None):
    """
        Adjust weights based on calculated gradients + learning rate scheduler, and refresh gradients
        Reference for optimizer_step: https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
        """
    if self.trainer.use_tpu:
      xm.optimizer_step(optimizer)
    else:
      # Adjust weights based on calculated gradients
      optimizer.step()

    # Refresh gradients (to zero)
    optimizer.zero_grad()
    # Update the learning rate scheduler
    self.lr_scheduler.step()

  # Step during validation

  def validation_step(self, batch, batch_idx):
    """
        Runs forward pass, calculates loss, and returns loss in a dict
        """

    # Return source and target text to calculate jaccard score only for validation
    loss, target_text = self._step(batch, return_text=True)

    preds = self.test_step(batch, batch_idx)
    preds_text = preds["preds"]
    # Track jaccard score to get validation accuracy
    jaccard_score = [jaccard(p, t) for p, t in zip(preds_text, target_text)]

    return {"val_loss": loss, "jaccard_score": jaccard_score}

  # Show loss after validation

  def validation_end(self, outputs):
    """
        Calculate average loss for all the validation batches
        """
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    jaccard_scores = sum([x["jaccard_score"] for x in outputs], [])
    avg_jaccard_score = np.mean(jaccard_scores)
    tensorboard_logs = {
        "val_loss": avg_loss,
        "jaccard_score": avg_jaccard_score
    }
    return {
        "avg_val_loss": avg_loss,
        "avg_jaccard_score": avg_jaccard_score,
        "log": tensorboard_logs
    }

  # Step during testing

  def test_step(self, batch, batch_idx):
    """
        Runs forward pass on test set and returns calculated loss, predictions, and targets
        Note: this assumes that your test set has targets (doesn't have for kaggle).
        """
    pad_token_id = self.tokenizer.pad_token_id
    source_ids, source_mask, _ = T5Dataset.trim_seq2seq_batch(batch,
                                                              pad_token_id,
                                                              test=True)
    # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_cnn.py
    # Generate reference: https://github.com/huggingface/transformers/blob/3e0f06210646a440509efa718b30d18322d6a830/src/transformers/modeling_utils.py#L769
    # For the sentiment span extraction task, turning off early stopping proved superior
    generated_ids = self.model.generate(
        input_ids=source_ids,
        attention_mask=source_mask,
        num_beams=1,
        max_length=80,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True,
    )
    preds = [
        self.tokenizer.decode(g,
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)
        for g in generated_ids
    ]

    return {"preds": preds}

  # Note: we don't attempt to print the loss from the test set, because it's assumed that we don't have the test targets
  def test_end(self, outputs):
    """
        """
    preds = []
    for pred in outputs:
      preds += pred["preds"]
    return {"preds": preds}

  def test_epoch_end(self, outputs):
    """
        Save test predictions and targets as text files and return the calculated loss for the test set
        """
    output_test_predictions_file = os.path.join(self.hparams.output_dir,
                                                "test_predictions.txt")
    # write predictions and targets for later rouge evaluation.
    with open(output_test_predictions_file, "w+") as p_writer:
      for output_batch in outputs:
        p_writer.writelines(s + "\n" for s in output_batch["preds"])
      p_writer.close()

    return self.test_end(outputs)

  def get_tqdm_dict(self):
    """
        Print average loss and learning rate at each step
        """
    avg_loss = getattr(self.trainer, "avg_loss", 0.0)
    tqdm_dict = {
        "loss": "{:.3f}".format(avg_loss),
        "lr": self.lr_scheduler.get_last_lr()[-1]
    }
    return tqdm_dict

  def _feature_file(self, mode):
    return os.path.join(
        self.hparams.data_dir,
        "cached_{}_{}_{}".format(
            mode,
            list(filter(None,
                        self.hparams.model_name_or_path.split("/"))).pop(),
            str(self.hparams.max_seq_length),
        ),
    )

  def is_logger(self):
    return self.trainer.proc_rank <= 0

  @staticmethod
  def add_model_specific_args(parser, root_dir):
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help=
        "Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)

    parser.add_argument(
        "--max_source_length",
        default=1024,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        default=56,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The input data dir. Should contain the dataset files for the text generation task.",
    )
    return parser


class LoggingCallback(pl.Callback):

  def on_validation_end(self, trainer: pl.Trainer,
                        pl_module: pl.LightningModule):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser, root_dir):
  parser.add_argument(
      "--output_dir",
      default=None,
      type=str,
      required=True,
      help=
      "The output directory where the model predictions and checkpoints will be written.",
  )

  parser.add_argument(
      "--fp16",
      action="store_true",
      help=
      "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
  )

  parser.add_argument(
      "--fp16_opt_level",
      type=str,
      default="O1",
      help=
      "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
      "See details at https://nvidia.github.io/apex/amp.html",
  )

  parser.add_argument("--n_gpu", type=int, default=1)
  parser.add_argument("--n_tpu_cores", type=int, default=0)
  parser.add_argument("--max_grad_norm",
                      default=1.0,
                      type=float,
                      help="Max gradient norm.")
  parser.add_argument("--do_train",
                      action="store_true",
                      help="Whether to run training.")
  parser.add_argument("--do_predict",
                      action="store_true",
                      help="Whether to run predictions on the test set.")
  parser.add_argument(
      "--gradient_accumulation_steps",
      type=int,
      default=1,
      help=
      "Number of updates steps to accumulate before performing a backward/update pass.",
  )

  parser.add_argument("--seed",
                      type=int,
                      default=42,
                      help="random seed for initialization")


def generic_train(model: T5Module, args: argparse.Namespace):
  # init model
  set_seed(args)

  if os.path.exists(args.output_dir) and os.listdir(
      args.output_dir) and args.do_train:
    raise ValueError(
        "Output directory ({}) already exists and is not empty.".format(
            args.output_dir))

  # Can take out checkpoint saving after each epoch to save memory
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
      gradient_clip_val=args.max_grad_norm,
      checkpoint_callback=checkpoint_callback,
      callbacks=[LoggingCallback()],
  )

  if args.fp16:
    train_params["use_amp"] = args.fp16
    train_params["amp_level"] = args.fp16_opt_level

  if args.n_tpu_cores > 0:
    global xm
    import torch_xla.core.xla_model as xm

    train_params["num_tpu_cores"] = args.n_tpu_cores
    train_params["gpus"] = 0

  if args.n_gpu > 1:
    train_params["distributed_backend"] = "ddp"

  trainer = pl.Trainer(**train_params)

  if args.do_train:
    trainer.fit(model)

  return trainer


# Finetuning
import argparse
import glob
import logging
import os
import time

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main(args):

  # If output_dir not provided, a folder will be generated in pwd
  if not args.output_dir:
    args.output_dir = os.path.join(
        "./results",
        f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(args.output_dir)
  model = T5Module(args)
  trainer = generic_train(model, args)

  # Save the last model as model.bin
  #checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
  #model = model.load_from_checkpoint(checkpoints[-1])
  model.model.save_pretrained(args.output_dir)
  # Save tokenizer files
  model.tokenizer.save_pretrained('./')

  # Optionally, predict on dev set and write to output_dir
  if args.do_predict:
    # See https://github.com/huggingface/transformers/issues/3159
    # pl use this format to create a checkpoint:
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/master\
    # /pytorch_lightning/callbacks/model_checkpoint.py#L169
    trainer.test(model)
  return trainer


# !mkdir / content / output

# Will set gpu on as soon as at least 1 batch works on cpu
# TODO: Consider factors here: https://github.com/huggingface/transformers/issues/3387
# Change LR to 1e-3 to 1e-4
#
ARGS_STR = """
--data_dir=/content/ \
--model_name_or_path=t5-base \
--learning_rate=3e-5 \
--train_batch_size=32 \
--output_dir=/content/output/ \
--do_train \
--n_gpu=1 \
--num_train_epochs 5 \
--max_source_length 80 \
"""
#
#--eval_batch_size=3 \
#--do_predict \

parser = argparse.ArgumentParser()
add_generic_args(parser, os.getcwd())
parser = T5Module.add_model_specific_args(parser, os.getcwd())
args = parser.parse_args(ARGS_STR.split())
trainer = main(args)

# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
# cat lightning_logs/version_0/hparams.yaml

# Test
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained('t5-base')
#t5 = T5ForConditionalGeneration.from_pretrained('/content/output/')
t5 = T5ForConditionalGeneration.from_pretrained('.')


def get_span(text):
  input_ids = tokenizer.encode(text,
                               return_tensors="pt",
                               add_special_tokens=True)  # Batch size 1
  t5.eval()
  generated_ids = t5.generate(
      input_ids=input_ids,
      num_beams=1,
      max_length=80,
      #repetition_penalty=2.5
  ).squeeze()
  predicted_span = tokenizer.decode(generated_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
  return predicted_span


get_span(
    "question: negative context: I`m in VA for the weekend, my youngest son turns 2 tomorrow......it makes me kinda sad, he is getting so big, check out my twipics"
)
get_span(
    "question: negative context: Recession hit Veronique Branquinho, she has to quit her company, such a shame!"
)
