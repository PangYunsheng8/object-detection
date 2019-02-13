"""Builder for dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from classification.utils import dataset_utils
from classification.protos import spec_pb2

slim = tf.contrib.slim

_FILE_PATTERN = '%s-*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = dataset_utils.ITEMS_TO_DESCRIPTIONS

def build(spec_config, is_training):
    """Builds a dataset based on the spec config.
    
    Arguments:
        spec_config: A spec.proto object containing the config for the desired
      dataset.
        is_training: True if this dataset is being built for training purposes.

    Returns:
      slim.dataset.Dataset based on the config.
    """
    if not isinstance(spec_config, spec_pb2.Spec):
        raise ValueError("spec_config not of type spec_pb2.Spec")
    dataset_dir = spec_config.dataset_dir
    num_classes = spec_config.num_classes
    split_name = "train" if is_training else "validation"
    return _get_split(split_name, dataset_dir, num_classes)

def _get_split(split_name, dataset_dir, num_classes, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading flowers.
    Args:
        split_name: A train/validation split name.
        dataset_dir: The base directory of the dataset sources.
        num_classes: Number of classes to predict.
        file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
        reader: The TensorFlow reader type.
    Returns:
        A `Dataset` namedtuple.
    Raises:
        ValueError: if `split_name` is not a valid train/validation split.
    """
    if split_name not in ["train", "validation"]:
        raise ValueError('split name should be train or validation, '
                         'but %s was given.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, split_name, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.load_label_map(dataset_dir)

    num_samples = dataset_utils.get_num_samples(dataset_dir, split_name)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=num_classes,
        labels_to_names=labels_to_names)
