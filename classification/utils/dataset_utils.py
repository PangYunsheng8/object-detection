"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json

from google.protobuf import text_format
import tensorflow as tf

from classification.protos import string_int_label_map_pb2

LABELS_FILENAME = 'label_map.pbtxt'

DISPLAY_NAME_FILENAME = "display_name_map.json"

DATASET_STATS_FILENAME = "dataset_stats.json"

FILE_PATTERN = '{:s}-{:05d}-of-{:05d}.tfrecord'

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}

def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
        values: A scalar or list of values.

    Returns:
        A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
        values: A string.

    Returns:
        A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    """Returns a TF-Feature of floats.

    Args:
        values: A scalar of list of values.

    Returns:
        A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))

def write_label_file(label_map_list, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

    Args:
        label_map_list: A list of dict consisting of 
        {'id':<id>,'class_name':<class_name>,'display_name':<display_name>}.
        dataset_dir: The directory in which the labels file should be written.
        filename: The filename where the class names are written.
    """
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    items = sorted(label_map_list, key=lambda x: int(x["id"]))
    labels_filename = os.path.join(dataset_dir, filename)
    for item_dict in items:
            item = label_map.item.add()
            item.id = item_dict["id"]
            item.name = item_dict["class"]
            item.display_name = item_dict["display_name"]
    with tf.gfile.Open(labels_filename, 'w') as fid:
        text_format.PrintMessage(label_map, fid)

def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

    Args:
        dataset_dir: The directory in which the labels file is found.
        filename: The filename where the class names are written.

    Returns:
        `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))

def load_label_map(dataset_dir, filename=LABELS_FILENAME):
    """Loads label map proto.

      Args:
        label_map_path: path to StringIntLabelMap proto text file.
      Returns:
        a StringIntLabelMapProto
      """
    label_map_path = os.path.join(dataset_dir, filename)
    with tf.gfile.GFile(label_map_path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    # validate label map
    for item in label_map.item:
        if item.id < 0:
            raise ValueError('Label map ids should be >= 0.')
    return label_map

def write_display_name_file(display_map, dataset_dir, 
                            filename=DISPLAY_NAME_FILENAME):
    """write display name map file into the dataset dir.

    Arguments:
        display_map: a map of id, class_names and display_names.
        dataset_dir: root dir of the dataset.
    """
    display_map.sort(key=lambda x: x["id"])
    dispaly_name_path = os.path.join(dataset_dir, filename)
    with tf.gfile.GFile(dispaly_name_path, 'w') as fid:
        json.dump(display_map, fid, indent=4)

def write_dataset_stats_file(dataset_stats, dataset_dir,
                             filename=DATASET_STATS_FILENAME):
    dataset_stats_path = os.path.join(dataset_dir, filename)
    with tf.gfile.GFile(dataset_stats_path, 'w') as fid:
        json.dump(dataset_stats, fid, indent=4, sort_keys=True)

def write_assets(dataset_dir, label_map_list, dataset_stats):
    write_label_file(label_map_list, dataset_dir)
    write_display_name_file(label_map_list, dataset_dir)
    write_dataset_stats_file(dataset_stats, dataset_dir)

def get_num_samples(dataset_dir, split_name,
                    filename=DATASET_STATS_FILENAME):
    dataset_stats_path = os.path.join(dataset_dir, filename)
    with tf.gfile.GFile(dataset_stats_path) as fid:
        dataset_stats = json.load(fid)
        return dataset_stats[split_name]