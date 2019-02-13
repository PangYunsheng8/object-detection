r"""Converts cropped image data to TFRecords of TF-Example protos.

This module recursivelly reads the cropped image files and creates 
two TFRecord datasets: one for train and one for test. 
Each TFRecord dataset is comprised of a set of TF-Example protocol 
buffers, each of which contain a single image and label.

The source images and labels information is given by a json file,
whose format is:
    {
        "classes":["foo", "bar"],
        "paths":[".../ex_1.jpg",".../ex_1.png"],
        "display_names":["biubiu", "xiuxiu"]
    }

The produced dataset format should be:
    root_dir
    label_map.pbtxt
    dispaly_name_map.json
    num_samples.json
    -- train
       -- train-00001-of-00010.tfrecord
       ...
    -- validation
       -- validation-00001-of-00005.tfrecord
       ...

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import io
import sys
import argparse
import json
from multiprocessing import Process
from PIL import Image
from collections import defaultdict

import tensorflow as tf
import tqdm
import numpy as np

from classification.utils import dataset_utils

_DISPLAY_NAME_FILENAME = dataset_utils.DISPLAY_NAME_FILENAME

_FILE_PATTERN = dataset_utils.FILE_PATTERN

_DATASET_STATS_FILENAME = dataset_utils.DATASET_STATS_FILENAME

_RANDOM_SEED = 0

def _get_args():
    parser = argparse.ArgumentParser(description="Convert image files to tfrecords.")
    parser.add_argument("--source_info_path", type=str, default="", 
                        help="source dir contains the image files.")
    parser.add_argument("--dataset_dir", type=str, default="", 
                        help="dataset dir contains the tfrecord files.")
    parser.add_argument("--num_shards", type=int, default=5,
                        help="output number of the tfrecord files.")
    parser.add_argument("--train_split_ratio", type=float, default=0.8, 
                        help="split ratio of training set.")
    args = parser.parse_args()
    return args

def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    output_filename = _FILE_PATTERN.format(
        split_name, shard_id+1, num_shards)
    return os.path.join(dataset_dir, split_name, output_filename)

def _build_single_cls_tfrecord(filenames_and_classes, output_filename, class_names_to_ids, process_bar):
    stats_to_collect = defaultdict(list)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for filename, class_name, display_name in filenames_and_classes:
            try:
                raw_image = Image.open(filename)
                width, height = raw_image.size
            except Exception as e:
                print(e)
                process_bar.update(1)
                continue

            stats_to_collect["classes"].append(class_name)
            stats_to_collect["paths"].append(filename)
            stats_to_collect["display_names"].append(display_name)

            _bytes = io.BytesIO()
            raw_image.save(_bytes, "jpeg")
            _bytes.seek(0)
            encoded_image = _bytes.read()

            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                encoded_image, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())
            process_bar.update(1)

    outjson_name = output_filename + "-info"
    with tf.gfile.GFile(outjson_name, "w") as fid:
        json.dump(stats_to_collect, fid, indent=4)


def _convert_dataset(split_name, filenames_and_classes, class_names_to_ids, num_shards, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames_and_classes: A list of tuples of absolute paths to png or jpg image and class name pair.
        class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
        dataset_dir: The directory where the converted datasets are stored.
        num_shards: Number of shards each split is comprised.
    """
    assert split_name in ['train', 'validation']

    split_dir = os.path.join(dataset_dir, split_name)
    if not tf.gfile.Exists(split_dir):
        tf.gfile.MakeDirs(split_dir)

    num_per_shard = int(math.ceil(len(filenames_and_classes) / float(num_shards)))
            
    processes = []
    for shard_id in range(num_shards):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, num_shards)
        
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id+1) * num_per_shard, len(filenames_and_classes))

        process_bar = tqdm.tqdm(total=end_ndx-start_ndx, 
                                desc=output_filename[-23:-9],
                                position=shard_id)
        p = Process(target=_build_single_cls_tfrecord, 
                    args=(filenames_and_classes[start_ndx:end_ndx],
                            output_filename,
                            class_names_to_ids,
                            process_bar))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def run(source_info_path, dataset_dir, num_shards, train_split_ratio):
    """Runs the download and conversion operation.

    Args:
        source_info_path: The json file contain raw data infomation.
        dataset_dir: The dataset directory where the dataset is stored.
        num_shards: The number of tfrecords each split contains.
        train_split_ratio: The split ratio of training set.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    
    with tf.gfile.FastGFile(source_info_path) as fid:
        source_info = json.load(fid)
        photo_filenames = source_info["paths"]
        class_names = source_info["classes"]
        display_names = source_info["display_names"]

    filename_and_classes = list(zip(photo_filenames, class_names, display_names))
    class_name_set = sorted(set(class_names))
    class_names_to_ids = dict(zip(class_name_set, range(len(class_name_set))))

    classes_to_indices = defaultdict(list)
    for idx, c in enumerate(class_names):
        classes_to_indices[c].append(idx)
    
    samples_stats = defaultdict(dict)
    training_indexes, validation_indexs = [], []
    for c, indexes in classes_to_indices.items():
        num_train = int(len(indexes) * train_split_ratio)
        training_indexes.extend(indexes[:num_train])
        validation_indexs.extend(indexes[num_train:])

        samples_stats[c]["num"] = len(indexes)
        samples_stats[c]["train"] = num_train
        samples_stats[c]["validation"] = len(indexes) - num_train

    # Divide into train and test:
    training_feeds = np.take(filename_and_classes, training_indexes, axis=0)
    validation_feeds = np.take(filename_and_classes, validation_indexs, axis=0)

    np.random.seed(_RANDOM_SEED)
    np.random.shuffle(training_feeds)
    np.random.shuffle(validation_feeds)
    
    # First, convert the training and validation sets.
    _convert_dataset('train', training_feeds, class_names_to_ids, num_shards,
                    dataset_dir)
    _convert_dataset('validation', validation_feeds, class_names_to_ids, num_shards,
                    dataset_dir)

    # Finally, write the labels file:
    dataset_stats = {
        "train": len(training_feeds),
        "validation": len(validation_feeds),
        "classes": dict(samples_stats)
    }
    
    dispaly_name_map = set(zip(class_names, display_names))
    dispaly_name_map = [{"id":class_names_to_ids[class_name],
                         "class":class_name, 
                         "display_name":display_name} for class_name, display_name in dispaly_name_map]
    dataset_utils.write_assets(dataset_dir, dispaly_name_map, dataset_stats)

if __name__ == "__main__":
    args = _get_args()
    run(args.source_info_path, args.dataset_dir, args.num_shards, args.train_split_ratio)
