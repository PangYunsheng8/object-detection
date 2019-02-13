r"""Evaluation executable for classification models.

This executable is used to evaluate classification models.

A pipeline_pb2.TrainEvalPipelineConfig file is specified.
In this mode, the --eval_training_data flag may be given to force the pipeline
to evaluate on training data instead.

Example usage:
    ./eval \
        --logtostderr \
        --pipeline_config_path=pipeline_config.pbtxt
"""
import os
import tensorflow as tf

from classification import evaluator
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
    eval_config = configs['eval_config']

    log_dir = spec_config.logdir
    eval_dir = os.path.join(log_dir, 'evaluation')
    tf.gfile.MakeDirs(eval_dir)
    tf.gfile.Copy(FLAGS.pipeline_config_path,
                os.path.join(eval_dir, 'pipeline.config'),
                overwrite=True)

    evaluator.evaluate(spec_config, 
                       eval_config)

if __name__ == '__main__':
  tf.app.run()
