"""Classification model evaluator.

This file provides a generic training method that can be used to train a
classification model.
"""
import os
import math

import tensorflow as tf

from classification.builders import preprocessor_builder
from classification.builders import model_builder
from classification.builders import dataset_builder
from classification.core import preprocessor
from classification.utils import visualization_utils as vsl_utils
from classification.utils import label_map_utils
from classification.utils import dataset_utils

slim = tf.contrib.slim

def get_inputs(dataset, num_classes, labels_offset,
               data_preprocess_options, eval_config):
    """get images and labels for evaluating.

    Args:
      dataset: a slim dataset.
      num_classes: number of classes to predict.
      labels_offset: an offset for the labels in the dataset.
      data_preprocess_options: a list of tuples, where each tuple contains a
        data preprocess function and a dictionary containing arguments and their
        values (see preprocessor.py).
      eval_config: a eval_pb2.EvalConfig protobuf.

    Returns:
      images: image tensors.
      labels: label tensors.
    """
    provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          shuffle=False,
          common_queue_capacity=2 * eval_config.batch_size,
          common_queue_min=eval_config.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= labels_offset

    if data_preprocess_options:
      image = preprocessor.preprocess(
          image, data_preprocess_options)

    images, labels = tf.train.batch(
            [image, label],
            batch_size=eval_config.batch_size,
            num_threads=eval_config.num_preprocessing_threads,
            capacity=5 * eval_config.batch_size)
   
    return images, labels

def _get_variables_to_restore(moving_average_decay):
    """Returns a list of variables to restore.
    Args:
      train_config: a eval_pb2.EvalConfig protobuf.
    Returns:
      A list of variables to restore from the checkpoint.
    """
    if moving_average_decay:
      global_step = slim.get_or_create_global_step()
      variable_averages = tf.train.ExponentialMovingAverage(
          moving_average_decay, global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[global_step.op.name] = global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()
    return variables_to_restore

def evaluate(spec_config, eval_config):
  """Evaluating function for classification models.

  Args:
    spec_config: a spec_pb2.Spec protobuf.
    eval_config: a train_pb2.TrainConfig protobuf.
  """
  num_classes = spec_config.num_classes
  labels_offset = spec_config.labels_offset
  moving_average_decay = spec_config.moving_average_decay

  data_preprocess_options = [
      preprocessor_builder.build(step)
      for step in eval_config.data_preprocess_options]

  # predictions_all, labels_all = [], []
  with tf.Graph().as_default():

      dataset = dataset_builder.build(spec_config, is_training=False)

      network_fn = model_builder.build(spec_config=spec_config,
                                      is_training=False)

      images, labels = get_inputs(dataset, num_classes, labels_offset,
                                  data_preprocess_options, eval_config)

      logits, _ = network_fn(images)

      variables_to_restore = _get_variables_to_restore(moving_average_decay)

      predictions = tf.argmax(logits, 1, name="predictions")
      labels = tf.squeeze(labels, name="labels")

      # Define the metrics:
      names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
          'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
          'Recall_5': slim.metrics.streaming_recall_at_k(
              logits, labels, 5),
          # 'eval_step': eval_step
      })

      # Print the summaries to screen.
      for name, value in names_to_values.items():
        summary_name = 'eval/%s' % name
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

      if eval_config.max_num_batches:
        num_batches = eval_config.max_num_batches
      else:
        # This ensures that we make a single pass over all of the data.
        num_batches = math.ceil(dataset.num_samples / float(eval_config.batch_size))
      
      checkpoint_dir = os.path.join(spec_config.logdir, "train")

      tf.logging.info('Evaluating %s' % checkpoint_dir)

      eval_dir = os.path.join(spec_config.logdir, "evaluation")
      tf.gfile.MakeDirs(eval_dir)
      
      label_map_path = os.path.join(spec_config.dataset_dir, dataset_utils.LABELS_FILENAME)
      display_names = label_map_utils.get_sorted_display_names(label_map_path)

      eval_op = names_to_updates
      eval_op["predictions"] = predictions
      eval_op["labels"] = labels
      eval_op["display_names"] = tf.constant(display_names)
      eval_op["global_step"] = slim.get_or_create_global_step()
      eval_op["summary_dir"] = tf.constant(eval_dir)

      slim.evaluation.evaluation_loop(
          master=eval_config.master,
          checkpoint_dir=checkpoint_dir,
          logdir=eval_dir,
          num_evals=num_batches,
          eval_op=eval_op,
          hooks=[vsl_utils.ConfusionMatrixHook()],
          variables_to_restore=variables_to_restore)
