"""Functions to build classification model training optimizers."""

import tensorflow as tf


def build(optimizer_config, global_summaries):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.
        global_summaries: A set to attach learning rate summary to.

    Returns:
        An optimizer.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    optimizer_type = optimizer_config.WhichOneof('optimizer')
    optimizer = None

    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        optimizer = tf.train.RMSPropOptimizer(
            _create_learning_rate(config.learning_rate, global_summaries),
            decay=config.decay,
            momentum=config.momentum_optimizer_value,
            epsilon=config.epsilon)

    if optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        optimizer = tf.train.MomentumOptimizer(
            _create_learning_rate(config.learning_rate, global_summaries),
            momentum=config.momentum_optimizer_value)

    if optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        optimizer = tf.train.AdamOptimizer(
            _create_learning_rate(config.learning_rate, global_summaries),
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon)

    if optimizer_type == 'adadelta_optimizer':
        config = optimizer_config.adadelta_optimizer
        optimizer = tf.train.AdadeltaOptimizer(
            _create_learning_rate(config.learning_rate, global_summaries),
            rho=config.rho,
            epsilon=config.epsilon)

    if optimizer_type == 'adagrad_optimizer':
        config = optimizer_config.adagrad_optimizer
        optimizer = tf.train.AdagradDAOptimizer(
            _create_learning_rate(config.learning_rate, global_summaries),
            intial_accumulator_value=config.intial_accumulator_value)

    if optimizer_type == 'ftrl_optimizer':
        config = optimizer_config.ftrl_optimizer
        optimizer = tf.train.FtrlOptimizer(
            _create_learning_rate(config.learning_rate, global_summaries),
            learning_rate_power=config.learning_rate_power,
            initial_accumulator_value=config.initial_accumulator_value,
            l1_regularization_strength=config.l1_regularization_strength,
            l2_regularization_strength=config.l2_regularization_strength)
    
    if optimizer_type == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(
            _create_learning_rate(config.learning_rate, global_summaries))

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    if optimizer_config.use_moving_average:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(
            optimizer, average_decay=optimizer_config.moving_average_decay)

    return optimizer


def _create_learning_rate(learning_rate_config, global_summaries):
    """Create optimizer learning rate based on config.

    Args:
        learning_rate_config: A LearningRate proto message.
        global_summaries: A set to attach learning rate summary to.

    Returns:
        A learning rate.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    learning_rate = None
    learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
    if learning_rate_type == 'constant_learning_rate':
        config = learning_rate_config.constant_learning_rate
        learning_rate = config.learning_rate

    if learning_rate_type == 'exponential_decay_learning_rate':
        config = learning_rate_config.exponential_decay_learning_rate
        learning_rate = tf.train.exponential_decay(
            config.initial_learning_rate,
            tf.train.get_or_create_global_step(),
            config.decay_steps,
            config.decay_factor,
            staircase=config.staircase)

    if learning_rate_type == 'polynomial_decay_earning_rate':
        config = learning_rate_config.polynomial_decay_earning_rate
        learning_rate = tf.train.polynomial_decay(
            config.initial_learning_rate,
            tf.train.get_or_create_global_step(),
            config.decay_steps,
            config.end_learning_rate,
            power=config.power,
            cycle=config.cycle)

    if learning_rate is None:
        raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

    global_summaries.add(tf.summary.scalar('Learning_Rate', learning_rate))
    return learning_rate
