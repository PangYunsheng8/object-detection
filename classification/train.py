r"""Training executable for classification models.

This executable is used to train classification models. 
A pipeline_pb2.TrainEvalPipelineConfig configuration file
can be specified by --pipeline_config_path.

Example usage:
    ./train \
        --logtostderr \
        --pipeline_config_path=pipeline_config.config
"""

import os
import tensorflow as tf

from classification import trainer
from classification.utils import config_util

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

FLAGS = flags.FLAGS

def main(_):
    assert FLAGS.pipeline_config_path, '`pipeline_config_path` is missing.'

    configs = config_util.get_configs_from_pipeline_file(
            FLAGS.pipeline_config_path)
    spec_config = configs['spec']
    train_config = configs['train_config']

    train_dir = os.path.join(spec_config.logdir, 'train')
    if train_config.task == 0: 
        tf.gfile.MakeDirs(train_dir)
        tf.gfile.Copy(FLAGS.pipeline_config_path,
                    os.path.join(train_dir, 'pipeline.config'),
                    overwrite=True)

    trainer.train(spec_config, train_config, train_dir)

if __name__ == '__main__':
  tf.app.run()
