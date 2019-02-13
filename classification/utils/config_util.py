"""Functions for reading and updating configuration files."""

import tensorflow as tf

from google.protobuf import text_format

from classification.protos import eval_pb2
from classification.protos import spec_pb2
from classification.protos import pipeline_pb2
from classification.protos import train_pb2


def get_configs_from_pipeline_file(pipeline_config_path):
  """Reads configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text
      proto.

  Returns:
    Dictionary of configuration objects. Keys are `spec`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`. Value are the
      corresponding config objects.
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

  configs = {}
  configs["spec"] = pipeline_config.spec
  configs["train_config"] = pipeline_config.train_config
  configs["eval_config"] = pipeline_config.eval_config

  return configs


def create_pipeline_proto_from_configs(configs):
  """Creates a pipeline_pb2.TrainEvalPipelineConfig from configs dictionary.

  This function nearly performs the inverse operation of
  get_configs_from_pipeline_file(). Instead of returning a file path, it returns
  a `TrainEvalPipelineConfig` object.

  Args:
    configs: Dictionary of configs. See get_configs_from_pipeline_file().

  Returns:
    A fully populated pipeline_pb2.TrainEvalPipelineConfig.
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  pipeline_config.spec.CopyFrom(configs["spec"])
  pipeline_config.train_config.CopyFrom(configs["train_config"])
  pipeline_config.eval_config.CopyFrom(configs["eval_config"])
  return pipeline_config


def get_number_of_classes(spec_config):
    """Returns the number of classes for a detection model.

    Args:
      model_config: A model_pb2.DetectionModel.

    Returns:
      Number of classes.

    Raises:
      ValueError: If the model type is not recognized.
    """

    return spec_config.num_classes


def get_optimizer_type(train_config):
    """Returns the optimizer type for training.

    Args:
      train_config: A train_pb2.TrainConfig.

    Returns:
      The type of the optimizer
    """
    return train_config.optimizer.WhichOneof("optimizer")


def get_learning_rate_type(optimizer_config):
    """Returns the learning rate type for training.

    Args:
      optimizer_config: An optimizer_pb2.Optimizer.

    Returns:
      The type of the learning rate.
    """
    return optimizer_config.learning_rate.WhichOneof("learning_rate")


def merge_external_params_with_configs(configs, hparams=None, **kwargs):
    """Updates `configs` dictionary based on supplied parameters.

    This utility is for modifying specific fields in the object detection configs.
    Say that one would like to experiment with different learning rates, momentum
    values, or batch sizes. Rather than creating a new config text file for each
    experiment, one can use a single base config file, and update particular
    values.

    Args:
      configs: Dictionary of configuration objects. See outputs from
        get_configs_from_pipeline_file() or get_configs_from_multiple_files().
      hparams: A `HParams`.
      **kwargs: Extra keyword arguments that are treated the same way as
        attribute/value pairs in `hparams`. Note that hyperparameters with the
        same names will override keyword arguments.

    Returns:
      `configs` dictionary.
    """

    if hparams:
      kwargs.update(hparams.values())
    for key, value in kwargs.items():
      if key == "learning_rate":
        _update_initial_learning_rate(configs, value)
        tf.logging.info("Overwriting learning rate: %f", value)
      if key == "batch_size":
        _update_batch_size(configs, value)
        tf.logging.info("Overwriting batch size: %d", value)
      if key == "momentum_optimizer_value":
        _update_momentum_optimizer_value(configs, value)
        tf.logging.info("Overwriting momentum optimizer value: %f", value)
      if key == "train_steps":
        _update_train_steps(configs, value)
        tf.logging.info("Overwriting train steps: %d", value)
      if key == "eval_steps":
        _update_eval_steps(configs, value)
        tf.logging.info("Overwriting eval steps: %d", value)
      if key == "dataset_dir":
        _update_dataset_dir(configs["dataset_dir"], value)
    return configs


def _update_initial_learning_rate(configs, learning_rate):
  """Updates `configs` to reflect the new initial learning rate.

  The configs dictionary is updated in place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    learning_rate: Initial learning rate for optimizer.

  Raises:
    TypeError: if optimizer type is not supported, or if learning rate type is
      not supported.
  """

  optimizer_type = get_optimizer_type(configs["train_config"])
  if optimizer_type == "rms_prop_optimizer":
    optimizer_config = configs["train_config"].optimizer.rms_prop_optimizer
  elif optimizer_type == "momentum_optimizer":
    optimizer_config = configs["train_config"].optimizer.momentum_optimizer
  elif optimizer_type == "adam_optimizer":
    optimizer_config = configs["train_config"].optimizer.adam_optimizer
  elif optimizer_type == "adadelta_optimizer":
    optimizer_config = configs["train_config"].optimizer.adadelta_optimizer
  elif optimizer_type == "adagrad_optimizer":
    optimizer_config = configs["train_config"].optimizer.adagrad_optimizer
  elif optimizer_type == "ftrl_optimizer":
    optimizer_config = configs["train_config"].optimizer.ftrl_optimizer
  else:
    raise TypeError("Optimizer %s is not supported." % optimizer_type)

  learning_rate_type = get_learning_rate_type(optimizer_config)
  if learning_rate_type == "constant_learning_rate":
    constant_lr = optimizer_config.learning_rate.constant_learning_rate
    constant_lr.learning_rate = learning_rate
  elif learning_rate_type == "exponential_decay_learning_rate":
    exponential_lr = (
        optimizer_config.learning_rate.exponential_decay_learning_rate)
    exponential_lr.initial_learning_rate = learning_rate
  elif learning_rate_type == "polynomial_decay_earning_rate":
    polynomial_lr = (
        optimizer_config.learning_rate.polynomial_decay_earning_rate)
    polynomial_lr.initial_learning_rate = learning_rate
  elif learning_rate_type == "manual_step_learning_rate":
    manual_lr = optimizer_config.learning_rate.manual_step_learning_rate
    original_learning_rate = manual_lr.initial_learning_rate
    learning_rate_scaling = float(learning_rate) / original_learning_rate
    manual_lr.initial_learning_rate = learning_rate
    for schedule in manual_lr.schedule:
      schedule.learning_rate *= learning_rate_scaling
  else:
    raise TypeError("Learning rate %s is not supported." % learning_rate_type)


def _update_batch_size(configs, batch_size):
  """Updates `configs` to reflect the new training batch size.

  The configs dictionary is updated in place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    batch_size: Batch size to use for training (Ideally a power of 2). Inputs
      are rounded, and capped to be 1 or greater.
  """
  configs["train_config"].batch_size = max(1, int(round(batch_size)))


def _update_momentum_optimizer_value(configs, momentum):
  """Updates `configs` to reflect the new momentum value.

  Momentum is only supported for RMSPropOptimizer and MomentumOptimizer. For any
  other optimizer, no changes take place. The configs dictionary is updated in
  place, and hence not returned.

  Args:
    configs: Dictionary of configuration objects. See outputs from
      get_configs_from_pipeline_file() or get_configs_from_multiple_files().
    momentum: New momentum value. Values are clipped at 0.0 and 1.0.

  Raises:
    TypeError: If the optimizer type is not `rms_prop_optimizer` or
    `momentum_optimizer`.
  """
  optimizer_type = get_optimizer_type(configs["train_config"])
  if optimizer_type == "rms_prop_optimizer":
    optimizer_config = configs["train_config"].optimizer.rms_prop_optimizer
  elif optimizer_type == "momentum_optimizer":
    optimizer_config = configs["train_config"].optimizer.momentum_optimizer
  else:
    raise TypeError("Optimizer type must be one of `rms_prop_optimizer` or "
                    "`momentum_optimizer`.")

  optimizer_config.momentum_optimizer_value = min(max(0.0, momentum), 1.0)

def _update_train_steps(configs, train_steps):
  """Updates `configs` to reflect new number of training steps."""
  configs["train_config"].num_steps = int(train_steps)


def _update_eval_steps(configs, eval_steps):
  """Updates `configs` to reflect new number of eval steps per evaluation."""
  configs["eval_config"].num_examples = int(eval_steps)


def _update_dataset_dir(spec_config, dataset_dir):
  """Updates spec configuration to reflect a new dataset dir."""
  spec_config["dataset_dir"] = dataset_dir

