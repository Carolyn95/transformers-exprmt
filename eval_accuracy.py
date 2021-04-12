"""Functions for evaluating accuracy for sentence classification."""
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def eval_accuracy(logits: np.ndarray,
                  label_ids: np.ndarray,
                  classes: list,
                  save_predictions=False) -> dict:
  """According to transformers.Trainer, this method must taka a :class:`~transformers.EvalPrediction` and return a dictionary string to metric values."""
  preds = logits[0] if isinstance(
      logits, tuple
  ) else logits  # take care of whether `output_hidden_states` or not in one line
  predictions = np.argmax(preds, axis=1)
  result = {
      'accuracy':
          accuracy_score(labels_ids, predictions),
      'report':
          classification_report(label_ids,
                                predictions,
                                digits=3,
                                target_names=classes)
  }
  if save_predictions:
    result['predictions'] = predictions
  return result


def get_compute_metrics(classes: list, save_predictions=False):

  def compute_metrics(p):
    return eval_accuracy(p.predictions,
                         p.label_ids,
                         classes,
                         save_predictions=save_predictions)

  return compute_metrics
