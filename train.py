"""Model training script."""
# from dataset_loaders.bankingAll import BankingAll
# from dataset_loaders.clincAll import ClincAll
from dataset_loaders.conll03 import Conll03
from sentence_classifier import SentenceClassifier
import pdb

DATASETS = {'conll': Conll03}

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      'model',
      type=str,
      help='Huggingface pretrained model or path to saved model on disk')
  parser.add_argument('dataset',
                      type=str,
                      help=f'Dataset to use {list(DATASETS)}')
  parser.add_argument('--out_dir', type=str, help='Directory to save model')
  parser.add_argument('--batch',
                      default=128,
                      help='Per GPU training batch size')
  parser.add_argument('--lr', default=1e-4, help='Learning rate')
  parser.add_argument('--epochs', default=10, help='Training epochs')
  args = parser.parse_args()
  # Run
  model = SentenceClassifier.create(args.model, DATASETS[args.dataset].load(),
                                    int(args.batch))
  if args.out_dir:
    model.args.output_dir = args.out_dir
  else:
    suffix = f'{args.dataset}_{args.epochs}epochs'
    model.args.output_dir = f'{model.args.output_dir}_{suffix}'
  model.args.num_train_epochs = int(args.epochs)
  model.args.learning_rate = float(args.lr)
  model.train(test_dataset=model.data['test'])
