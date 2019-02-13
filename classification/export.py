# encoding=utf-8

import os
import json

import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib

from classification.builders import model_builder
from classification.builders import preprocessor_builder
from classification.core import preprocessor
from classification.utils import config_util
from classification.utils import dataset_utils

flags = tf.app.flags
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string("export_dir", "",
                    "Export directory")
flags.DEFINE_string("tensor_prefix", "",
                    "The prefix of input and output tensor")
FLAGS = flags.FLAGS

def freeze_graph_with_def_protos(
        input_graph_def,
        input_saver_def,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        clear_devices,
        initializer_nodes,
        optimize_graph=True,
        variable_names_blacklist=''):
    """Converts all variables in a graph and checkpoint into constants."""
    del restore_op_name, filename_tensor_name  # Unused by updated loading code.

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if not saver_lib.checkpoint_exists(input_checkpoint):
        raise ValueError(
            'Input checkpoint "' + input_checkpoint + '" does not exist!')

    if not output_node_names:
        raise ValueError(
            'You must supply the name of a node to --output_node_names.')

    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ''

    with tf.Graph().as_default():
        tf.import_graph_def(input_graph_def, name='')

        if optimize_graph:
            rewrite_options = rewriter_config_pb2.RewriterConfig(
                optimize_tensor_layout=True)
            rewrite_options.optimizers.append('pruning')
            rewrite_options.optimizers.append('constfold')
            rewrite_options.optimizers.append('layout')
            graph_options = tf.GraphOptions(
                rewrite_options=rewrite_options, infer_shapes=True)
        else:
            graph_options = tf.GraphOptions()
        config = tf.ConfigProto(graph_options=graph_options)
        with session.Session(config=config) as sess:
            if input_saver_def:
                saver = saver_lib.Saver(saver_def=input_saver_def)
                saver.restore(sess, input_checkpoint)
            else:
                var_list = {}
                reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
                var_to_shape_map = reader.get_variable_to_shape_map()
                for key in var_to_shape_map:
                    try:
                        tensor = sess.graph.get_tensor_by_name(key + ':0')
                    except KeyError:
                        # This tensor doesn't exist in the graph (for example it's
                        # 'global_step' or a similar housekeeping element) so skip it.
                        continue
                    var_list[key] = tensor
                saver = saver_lib.Saver(var_list=var_list)
                saver.restore(sess, input_checkpoint)
                if initializer_nodes:
                    sess.run(initializer_nodes)

            variable_names_blacklist = (variable_names_blacklist.split(',') if
                                        variable_names_blacklist else None)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(','),
                variable_names_blacklist=variable_names_blacklist)

    return output_graph_def


def _write_saved_model(saved_model_path,
                       frozen_graph_def,
                       inputs,
                       outputs):
    """Writes SavedModel to disk.

    If checkpoint_path is not None bakes the weights into the graph thereby
    eliminating the need of checkpoint files during inference. If the model
    was trained with moving averages, setting use_moving_averages to true
    restores the moving averages, otherwise the original set of variables
    is restored.

    Args:
      saved_model_path: Path to write SavedModel.
      frozen_graph_def: tf.GraphDef holding frozen graph.
      inputs: The input image tensor to use for detection.
      outputs: A tensor dictionary containing the outputs of a DetectionModel.
    """
    with tf.Graph().as_default():
        with session.Session() as sess:
            tf.import_graph_def(frozen_graph_def, name='')

            if gfile.IsDirectory(saved_model_path):
                gfile.DeleteRecursively(saved_model_path)
            builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)

            tensor_info_inputs = {
                'inputs': tf.saved_model.utils.build_tensor_info(inputs)}
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(v)

            detection_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=tensor_info_inputs,
                    outputs=tensor_info_outputs,
                    method_name=signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        detection_signature,
                },
            )
            builder.save()


def get_tf_version():
    import re
    import tensorflow as tf
    vs = tf.__version__
    vs = re.findall("^(\d+)\.(\d+)", vs)[0]
    vs = [int(x) for x in vs]
    return vs


def main(_):
    assert FLAGS.pipeline_config_path, '`pipeline_config_path` is missing.'

    configs = config_util.get_configs_from_pipeline_file(
            FLAGS.pipeline_config_path)
    spec_config = configs['spec']
    eval_config = configs['eval_config']

    data_preprocess_options = [
      preprocessor_builder.build(step)
      for step in eval_config.data_preprocess_options]

    graph = tf.Graph()
    with graph.as_default():
        input_images = tf.placeholder(tf.uint8, shape=[None, None, None, 3],
                                      name=FLAGS.tensor_prefix + "input_images")

        network_fn = model_builder.build(spec_config,
                                        is_training=False)

        def preprocess_fn(image):
            return preprocessor.preprocess(
                        image, data_preprocess_options)
        
        input_images = tf.map_fn(preprocess_fn, 
                               input_images, 
                               dtype=tf.float32,
                               parallel_iterations=8,
                               back_prop=False)

        logits, edp = network_fn(input_images)

        outputs = {}
        outputs[FLAGS.tensor_prefix + "predict"] = \
            tf.argmax(logits, axis=-1, name=FLAGS.tensor_prefix + "predict")
        outputs[FLAGS.tensor_prefix + "logits"] = \
            tf.identity(logits, name=FLAGS.tensor_prefix + "logits")
        outputs[FLAGS.tensor_prefix + "features"] = \
            tf.nn.softmax(logits, name=FLAGS.tensor_prefix + "features")

        sv_def = tf.train.Saver().as_saver_def()
        gap_def = graph.as_graph_def()
    tf_version = get_tf_version()
    if tf_version[0] < 0 or tf_version == 1 and tf_version[1] <= 4:
        optimize_graph = True
    else:
        optimize_graph = False
    output_node_names = ",".join(list(outputs.keys()))
    checkpoint_path = tf.train.latest_checkpoint(
                    os.path.join(spec_config.logdir, "train"))
    freeze_graph_def = freeze_graph_with_def_protos(
        input_graph_def=gap_def,
        input_saver_def=sv_def,
        input_checkpoint=checkpoint_path,
        output_node_names=output_node_names,
        restore_op_name="", filename_tensor_name="",
        clear_devices=True,
        initializer_nodes=False,
        optimize_graph=optimize_graph
    )

    gfile.MakeDirs(FLAGS.export_dir)
    with gfile.GFile(os.path.join(FLAGS.export_dir, "frozen_inference_graph.pb"), "wb") as f:
        f.write(freeze_graph_def.SerializeToString())

    _write_saved_model(os.path.join(FLAGS.export_dir, "saved_model"),
                       freeze_graph_def,
                       inputs=input_images,
                       outputs=outputs)

    src_label_map_path = os.path.join(spec_config.dataset_dir, 
                                      dataset_utils.DISPLAY_NAME_FILENAME)
    dst_label_map_path = os.path.join(FLAGS.export_dir, "label_index.map")
    tf.gfile.Copy(src_label_map_path, dst_label_map_path)
    with open(os.path.join(FLAGS.export_dir, "model_info.json"), "w") as f:
        model_info = {
            "input_output_tensor_prefix": FLAGS.tensor_prefix
        }
        json.dump(model_info, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    tf.app.run()
