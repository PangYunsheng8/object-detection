"""Classification model trainer.

This file provides a generic training method that can be used to train a
classification model.
"""

import tensorflow as tf

from classification.builders import optimizer_builder
from classification.builders import preprocessor_builder
from classification.builders import model_builder
from classification.builders import dataset_builder
from classification.core import preprocessor
from classification.core import model_deploy
from classification.utils import variables_helper

slim = tf.contrib.slim

def create_input_queue(batch_size_per_clone, dataset,
                       num_classes, labels_offset,
                       data_augmentation_options, train_config):
    """batch, prefetch and returns input queue.

    Args:
      batch_size_per_clone: batch size to use per clone.
      dataset: a slim dataset.
      num_classes: number of classes to predict.
      labels_offset: an offset for the labels in the dataset.
      data_augmentation_options: a list of tuples, where each tuple contains a
        data augmentation function and a dictionary containing arguments and their
        values (see preprocessor.py).
      train_config: a train_pb2.TrainConfig protobuf.

    Returns:
      input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
        (which hold images, boxes and targets).  To get a batch of tensor_dicts,
        call input_queue.Dequeue().
    """
    provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=train_config.num_readers,
          common_queue_capacity=20 * train_config.batch_size,
          common_queue_min=10 * train_config.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= labels_offset

    if data_augmentation_options:
      image = preprocessor.preprocess(
          image, data_augmentation_options)

    images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size_per_clone,
            num_threads=train_config.num_batch_queue_threads,
            capacity=train_config.batch_queue_capacity)
    labels = slim.one_hot_encoding(
        labels, num_classes - labels_offset)
    input_queue = slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=train_config.prefetch_queue_capacity)

    return input_queue

def clone_fn(input_queue, network_fn, train_config):
    """Allows data parallelism by creating multiple clones of network_fn.

    Args:
      input_queue: input queue holding enqueued tensor_dicts.
      network_fn: classification model function.
      train_config: a train_pb2.TrainConfig protobuf.
    """
    images, labels = input_queue.dequeue()
    logits, end_points = network_fn(images)

    if 'AuxLogits' in end_points:
      slim.losses.softmax_cross_entropy(
          end_points['AuxLogits'], labels,
          label_smoothing=train_config.label_smoothing, weights=0.4,
          scope='aux_loss')
    slim.losses.softmax_cross_entropy(
        logits, labels, label_smoothing=train_config.label_smoothing, weights=1.0)
    return end_points

def _get_init_fn(train_config, train_dir):
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.
    Args:
      train_config: a train_pb2.TrainConfig protobuf.
      train_dir: Directory write checkpoints and summaries to.
    Returns:
      An init function run by the supervisor.
    """
    if train_config.fine_tune_checkpoint is None:
      return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % train_dir)
        return None

    exclusions = []
    if train_config.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                    for scope in train_config.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
      for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
          break
      else:
        variables_to_restore.append(var)

    if tf.gfile.IsDirectory(train_config.fine_tune_checkpoint):
      checkpoint_path = tf.train.latest_checkpoint(train_config.fine_tune_checkpoint)
    else:
      checkpoint_path = train_config.fine_tune_checkpoint

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=train_config.ignore_missing_vars)

def _get_variables_to_train(train_config):
    """Returns a list of variables to train.
    Args:
      train_config: a train_pb2.TrainConfig protobuf.
    Returns:
      A list of variables to train by the optimizer.
    """
    if train_config.trainable_scopes is None:
      return tf.trainable_variables()
    else:
      scopes = [scope.strip() for scope in train_config.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
      variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
      variables_to_train.extend(variables)
    return variables_to_train

def train(spec_config,
          train_config,
          train_dir):
  """Training function for classification models.

  Args:
    spec_config: a spec_pb2.Spec protobuf.
    train_config: a train_pb2.TrainConfig protobuf.
    train_dir: Directory write checkpoints and summaries to.
  """
  num_classes = spec_config.num_classes
  labels_offset = spec_config.labels_offset
  moving_average_decay = spec_config.moving_average_decay

  data_augmentation_options = [
      preprocessor_builder.build(step)
      for step in train_config.data_augmentation_options]

  with tf.Graph().as_default():
    # Build a configuration specifying multi-GPU and multi-replicas.
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=train_config.num_clones,
        clone_on_cpu=train_config.clone_on_cpu,
        replica_id=train_config.task,
        num_replicas=train_config.worker_replicas,
        num_ps_tasks=train_config.num_ps_tasks)

    # Place the global step on the device storing the variables.
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    dataset = dataset_builder.build(spec_config, is_training=True)

    network_fn = model_builder.build(spec_config=spec_config,
                                     is_training=True,
                                     weight_decay=train_config.weight_decay)

    with tf.device(deploy_config.inputs_device()):
      
      input_queue = create_input_queue(
          train_config.batch_size // train_config.num_clones,
          dataset, num_classes, labels_offset,
          data_augmentation_options, train_config)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_summaries = set([])

    clones = model_deploy.create_clones(deploy_config, 
                                        clone_fn, 
                                        [input_queue, network_fn, train_config])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
    
    #################################
    # Configure the moving averages #
    #################################
    if moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      training_optimizer = optimizer_builder.build(train_config.optimizer,
                                                   global_summaries)
    sync_optimizer = None
    if train_config.sync_replicas:
      training_optimizer = tf.train.SyncReplicasOptimizer(
          training_optimizer,
          replicas_to_aggregate=train_config.replicas_to_aggregate,
          total_num_replicas=train_config.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
      sync_optimizer = training_optimizer
    elif moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    variables_to_train = _get_variables_to_train(train_config)

    with tf.device(deploy_config.optimizer_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, training_optimizer, var_list=variables_to_train)
      total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')

      # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
      if train_config.bias_grad_multiplier:
        biases_regex_list = ['.*/biases']
        grads_and_vars = variables_helper.multiply_gradients_matching_regex(
            grads_and_vars,
            biases_regex_list,
            multiplier=train_config.bias_grad_multiplier)

      # Optionally freeze some layers by setting their gradients to be zero.
      if train_config.freeze_variables:
        grads_and_vars = variables_helper.freeze_gradients_matching_regex(
            grads_and_vars, train_config.freeze_variables)

      # Optionally clip gradients
      if train_config.gradient_clipping_by_norm > 0:
        with tf.name_scope('clip_grads'):
          grads_and_vars = slim.learning.clip_gradient_norms(
              grads_and_vars, train_config.gradient_clipping_by_norm)

      # Create gradient updates.
      grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                        global_step=global_step)
      update_ops.append(grad_updates)

      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')
    
    ####################################################
    #                Add summaries                     #
    ####################################################

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    for model_var in slim.get_model_variables():
      global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
    global_summaries.add(
        tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    slim.learning.train(
        train_tensor,
        logdir=train_dir,
        master=train_config.master,
        is_chief=(train_config.task == 0),
        session_config=session_config,
        init_fn=_get_init_fn(train_config, train_dir),
        summary_op=summary_op,
        number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        log_every_n_steps=train_config.log_every_n_steps,
        save_summaries_secs=train_config.save_summaries_secs,
        save_interval_secs=train_config.save_interval_secs,
        sync_optimizer=sync_optimizer,
        saver=saver)
