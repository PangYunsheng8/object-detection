# encoding=utf-8

import os
import math
import json

import tqdm
from lxml import etree
from PIL import Image
from PIL import ImageFile
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

import od_example
import string_int_label_map_pb2

ImageFile.LOAD_TRUNCATED_IMAGES = True

Flags = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("output_path", "",
                           "Output tfrecord path")
tf.app.flags.DEFINE_string("dataset_dir", "",
                           "Dataset directory witch contain image and its annotation")
tf.app.flags.DEFINE_string("label_map_path", "",
                           "Object name to label map file path")
tf.app.flags.DEFINE_string("class_name_map_path", "",
                           "Object class name to an uniform label name map file path")
tf.app.flags.DEFINE_string("image_folder", "",
                           "Image's folder. default: ''")
tf.app.flags.DEFINE_string("annotation_folder", "",
                           "Annotation's folder. default: ''")
tf.app.flags.DEFINE_string("mask_folder", "",
                           "Instance Mask's folder. default: ''")
tf.app.flags.DEFINE_string("split_by", "output_number",
                           "Split tfrecord by 'output_number' or 'max_example'. default: 'output_number'")
tf.app.flags.DEFINE_integer("arg_number", 1,
                            "Split operation's argument number. default: 1")


def _recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = _recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def _read_annotation(annotation_path):
    with tf.gfile.GFile(annotation_path, 'rb') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = _recursive_parse_xml_to_dict(xml)["annotation"]
    return data


def _load_label_map(label_map_path):
    """Loads label map proto.

      Args:
        label_map_path: path to StringIntLabelMap proto text file.
      Returns:
        a StringIntLabelMapProto
      """
    with tf.gfile.GFile(label_map_path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    # validate label map
    for item in label_map.item:
        if item.id < 1:
            raise ValueError('Label map ids should be >= 1.')
    return label_map


def _load_label_map_dict(label_map_path):
    label_map = _load_label_map(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        label_map_dict[item.name] = item.id
    return label_map_dict


def _save_label_map(label_map, label_map_path):
    """Save label map proto.

      Args:
        label_map: StringIntLabelMap proto object
        label_map_path: path to StringIntLabelMap proto text file.
      Returns:
        a StringIntLabelMapProto
      """
    with tf.gfile.GFile(label_map_path, 'w') as fid:
        text_format.PrintMessage(label_map, fid)


def _save_label_map_dict(label_map_dict, label_map_path):
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    items = sorted(label_map_dict.items(), key=lambda x: int(x[1]))
    for name, id_ in items:
        item = label_map.item.add()
        item.id = id_
        item.name = name
    _save_label_map(label_map, label_map_path)


def _get_instance_mask_map():
    import cv2
    folder = "/home/zhaojh/PycharmProjects/cola/data/front"
    instance_masks = {}
    for sub_folder in sorted(os.listdir(folder)):
        sub_folder_ = os.path.join(folder, sub_folder)
        for name in sorted(os.listdir(sub_folder_)):
            if sub_folder not in instance_masks:
                instance_masks[sub_folder] = []
            front_img_path = os.path.join(sub_folder_, name)
            im = cv2.imread(front_img_path, cv2.IMREAD_UNCHANGED)
            mask = np.asarray(im[:, :, -1] > 64, np.bool)
            instance_masks[sub_folder].append(mask)
    return instance_masks


def create_od_tfrecord(path, dataset_dir,
                       label_map_path="",
                       class_name_map_path="",
                       image_folder="images",
                       annotation_folder="annotations",
                       mask_folder="mask",
                       split_by_output_number=True,
                       arg_number=1):
    label_map_dict = {}
    if label_map_path:
        label_map_dict = _load_label_map_dict(label_map_path)
    class_name_map_dict = {}
    if class_name_map_path:
        with open(class_name_map_path) as f:
            class_name_map_dict = json.load(f)

    instance_mask_map = _get_instance_mask_map()

    images_dir = os.path.join(dataset_dir, image_folder)
    annotation_dir = os.path.join(dataset_dir, annotation_folder)
    mask_dir = os.path.join(dataset_dir, mask_folder)

    image_names = [_ for _ in os.listdir(images_dir) if _.lower().rsplit(".", 1)[-1] in ('jpg', 'jpeg', 'png')]
    image_count = len(image_names)
    print("Found %s images in %s" % (image_count, images_dir))

    if split_by_output_number:
        arg_number = int(math.ceil(image_count / float(arg_number)))
    patch_names = [image_names[i:i + arg_number] for i in range(0, image_count - 1, arg_number)]

    tfrecord_dir = os.path.dirname(path)
    if not os.path.exists(tfrecord_dir):
        tf.gfile.MakeDirs(tfrecord_dir)

    process_bar = tqdm.tqdm(total=image_count)
    for i, names in enumerate(patch_names):
        t = path.rsplit(".", 1)
        if len(t) == 1:
            record_path = path + "_%s" % i
        else:
            record_path = t[0] + "_%s." % i + t[1]
        writer = tf.python_io.TFRecordWriter(record_path)

        for image_name in names:
            name = image_name.rsplit(".", 1)[0]
            annotation_name = name + ".xml"
            image_path = os.path.join(images_dir, image_name)
            annotation_path = os.path.join(annotation_dir, annotation_name)
            # read annotation
            one_image_data = _read_annotation(annotation_path)
            for j, obj in enumerate(one_image_data["object"]):
                # adjust class name
                obj["name"] = obj["name"].strip()
                if obj["name"] in class_name_map_dict:
                    obj["name"] = class_name_map_dict.get(obj["name"])
                if obj["name"] not in label_map_dict:
                    label_map_dict[obj["name"]] = len(label_map_dict) + 1
                    print("Add new item {id: %s, name: %s} to label map." % (len(label_map_dict), obj["name"]))
                one_image_data["object"][j]["label"] = label_map_dict[obj["name"]]
                one_image_data["object"][j]["text"] = obj["name"]
            # read image
            im = Image.open(image_path)
            im_arr = np.asarray(im)
            one_image_data["image_data"] = im_arr
            # read mask
            mask_path = os.path.join(mask_dir, name + ".mask")
            with open(mask_path) as f:
                masks = json.load(f)
                masks = [instance_mask_map[ms[0]][ms[1]] for ms in masks]
                corners = ["ymin", "xmin", "ymax", "xmax"]
                boxes = [[int(obj["bndbox"][cn]) for cn in corners] for obj in one_image_data["object"]]
                instance_masks = []
                for ms, (ymin, xmin, ymax, xmax) in zip(masks, boxes):
                    ms = Image.fromarray(ms.astype(np.uint8)).resize((xmax - xmin, ymax - ymin))
                    ms = np.asarray(ms)
                    ms = ms.astype(np.float32)
                    ept = np.zeros(im_arr.shape[:2], dtype=np.float32)
                    ept[ymin:ymax, xmin:xmax] = ms
                    instance_masks.append(ept)
            one_image_data["instance_masks"] = instance_masks

            one_image_data = od_example.create_one_od_example(one_image_data)
            writer.write(one_image_data.SerializeToString())
            process_bar.update(1)

        writer.close()
        print("Wrote %s examples to file (%s)" % (len(names), record_path))

    output_label_map_path = os.path.join(tfrecord_dir, "label_map.pbtxt")
    _save_label_map_dict(label_map_dict, output_label_map_path)


def main(_):
    assert Flags.output_path != ""
    assert Flags.arg_number > 0

    if Flags.split_by.lower() == "output_number":
        print("Split TFRecord by output file number [%s]." % Flags.arg_number)
        split_by_output_number = True
    else:
        print("Split TFRecord by max examples in one file [%s]." % Flags.arg_number)
        split_by_output_number = False

    create_od_tfrecord(Flags.output_path, Flags.dataset_dir, Flags.label_map_path,
                       Flags.class_name_map_path, Flags.image_folder, Flags.annotation_folder,
                       Flags.mask_folder, split_by_output_number, Flags.arg_number)


if __name__ == '__main__':
    tf.app.run()
