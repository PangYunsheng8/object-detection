# encoding=utf-8

import io
import hashlib

from PIL import Image
import numpy as np
import tensorflow as tf


def _int64_list_feature(value):
    if isinstance(value, (list, tuple)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(value):
    if isinstance(value, (list, tuple)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    if isinstance(value, (list, tuple)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def create_one_od_example(image_data, max_edge=2048):
    """ Create TFRecord example
    :param image_data: dict, an image_data example
        {
            "filename": "test.jpg",
            "source_id": "test.jpg",
            "image_data": np.ones((28,28,1), dtype=np.uint8),
            "format": "JPEG",
            "object":[
                {
                    "bndbox":{"xmin": 0., "xmax": 1., "ymin": 0., "ymax": 1.},
                    "label": 1,
                    "text": 'bird',
                    "difficult": 0,
                    "truncated": 0,
                    "pose": ''
                },
                ...]
        }
    :return: a tf.train.Example
    """
    _image_arr = np.asarray(image_data["image_data"], dtype=np.uint8)
    _img = Image.fromarray(_image_arr)
    width_, height_ = _img.size
    _p = max(width_, height_) / max_edge
    if _p > 1:
        _img = _img.resize((int(round(width_ / _p)), int(round(height_ / _p))))
    _bio = io.BytesIO()
    _img.save(_bio, "jpeg")
    _bio.seek(0, 0)
    width, height = _img.size
    encoded_img = _bio.read()
    key = hashlib.sha256(encoded_img).hexdigest()

    xmin, xmax, ymin, ymax = [], [], [], []
    classes, classes_text, difficult_obj, truncated, poses = [], [], [], [], []
    for obj in image_data.get("object", []):
        if float(obj["bndbox"]["xmin"]) > width_ or \
                        float(obj["bndbox"]["xmax"]) > width_ or \
                        float(obj["bndbox"]["ymin"]) > height_ or \
                        float(obj["bndbox"]["ymax"]) > height_:
            print("remove illegal object", obj["bndbox"])
            continue
        xmin.append(float(obj["bndbox"]["xmin"]) / width_)
        xmax.append(float(obj["bndbox"]["xmax"]) / width_)
        ymin.append(float(obj["bndbox"]["ymin"]) / height_)
        ymax.append(float(obj["bndbox"]["ymax"]) / height_)
        classes.append(int(obj["label"]))
        classes_text.append(obj.get("text", '').encode('utf8'))
        difficult_obj.append(int(obj.get("difficult", 0)))
        truncated.append(int(obj.get("truncated", 0)))
        poses.append(obj.get("pose", '').encode('utf8'))

    feature = {
        'image/height': _int64_list_feature(height),
        'image/width': _int64_list_feature(width),
        'image/filename': _bytes_list_feature(image_data.get("filename", "").encode('utf8')),
        'image/source_id': _bytes_list_feature(image_data.get("source_id", "").encode('utf8')),
        'image/key/sha256': _bytes_list_feature(key.encode('utf8')),
        'image/encoded': _bytes_list_feature(encoded_img),
        'image/format': _bytes_list_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': _float_list_feature(xmin),
        'image/object/bbox/xmax': _float_list_feature(xmax),
        'image/object/bbox/ymin': _float_list_feature(ymin),
        'image/object/bbox/ymax': _float_list_feature(ymax),
        'image/object/class/text': _bytes_list_feature(classes_text),
        'image/object/class/label': _int64_list_feature(classes),
        'image/object/difficult': _int64_list_feature(difficult_obj),
        'image/object/truncated': _int64_list_feature(truncated),
        'image/object/view': _bytes_list_feature(poses),
    }

    if "instance_masks" in image_data:
        instance_masks = image_data["instance_masks"]
        for idx, mask in enumerate(instance_masks):
            if (width_, height_) != (width, height):
                mask = Image.fromarray(mask.astype(np.uint8))
                mask = mask.resize((width, height))
                instance_masks[idx] = np.asarray(mask).astype(np.bool)
        instance_masks = np.stack(instance_masks)
        indices = np.where(instance_masks)
        indices = np.asarray(indices).transpose()

        feature['image/object/mask/indices'] = _int64_list_feature(indices.flatten().tolist()),
        feature['image/object/mask/values'] = _float_list_feature(np.ones(len(indices)).tolist()),
        feature['image/object/mask/shape'] = _int64_list_feature(instance_masks.shape)

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example


def parse_one_od_example(serialized_example, parse_mask=False):
    """ Parse one serialized object detection example
    :param serialized_example:
    :param parse_mask: parse mask or not
    :return:
    """
    features = {
        'image/height': tf.FixedLenFeature((), tf.int64),
        'image/width': tf.FixedLenFeature((), tf.int64),
        'image/filename': tf.FixedLenFeature((), tf.string),
        'image/source_id': tf.FixedLenFeature((), tf.string),
        'image/key/sha256': tf.FixedLenFeature((), tf.string),
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/object/difficult': tf.VarLenFeature(tf.int64),
        'image/object/truncated': tf.VarLenFeature(tf.int64),
        'image/object/view': tf.VarLenFeature(tf.string),
    }
    if parse_mask:
        features['image/object/mask/indices'] = tf.VarLenFeature(tf.int64)
        features['image/object/mask/values'] = tf.VarLenFeature(tf.float32)
        features['image/object/mask/shape'] = tf.VarLenFeature(tf.int64)

    example = tf.parse_single_example(serialized=serialized_example, features=features)
    example["image/encoded"] = tf.image.decode_jpeg(example["image/encoded"], channels=3)
    example["image/object/bbox/xmin"] = tf.sparse_tensor_to_dense(example["image/object/bbox/xmin"])
    example["image/object/bbox/xmax"] = tf.sparse_tensor_to_dense(example["image/object/bbox/xmax"])
    example["image/object/bbox/ymin"] = tf.sparse_tensor_to_dense(example["image/object/bbox/ymin"])
    example["image/object/bbox/ymax"] = tf.sparse_tensor_to_dense(example["image/object/bbox/ymax"])
    example["image/object/class/text"] = tf.sparse_tensor_to_dense(example["image/object/class/text"], "")
    example["image/object/class/label"] = tf.sparse_tensor_to_dense(example["image/object/class/label"])
    example["image/object/difficult"] = tf.sparse_tensor_to_dense(example["image/object/difficult"])
    example["image/object/truncated"] = tf.sparse_tensor_to_dense(example["image/object/truncated"])
    example["image/object/view"] = tf.sparse_tensor_to_dense(example["image/object/view"], "")
    if parse_mask:
        indices = tf.sparse_tensor_to_dense(example["image/object/mask/indices"])
        values = tf.sparse_tensor_to_dense(example["image/object/mask/values"])
        shape = tf.sparse_tensor_to_dense(example["image/object/mask/shape"])
        mask = tf.SparseTensor(indices, values, shape)
        example["image/object/mask"] = tf.sparse_tensor_to_dense(mask)

    return example
