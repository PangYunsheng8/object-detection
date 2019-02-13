# encoding=utf-8

"""
Visualize image ground truth bbox

"""

import os
from lxml import etree

import tqdm
from PIL import Image
import numpy as np
import tensorflow as tf
from PIL import ImageFile

import sys
import requests

api_url = "http://47.92.9.46/label/name/"

ImageFile.LOAD_TRUNCATED_IMAGES = True

prj_fld = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = os.path.join(prj_fld, "detection", "object_detection", "utils")
print(path)
if path not in sys.path:
    sys.path.append(path)
import visualization_utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("images_dir", "",
                           "image directory")
tf.app.flags.DEFINE_string("annotations_dir", "",
                           "annotations directory")
tf.app.flags.DEFINE_string("outputs_dir", "",
                           "outputs directory")
tf.app.flags.DEFINE_bool("cvt_uuid_to_pinyin", False,
                         "whether convert uuid to more readable pinyin")


def request_sku_info(uuid, retry=3):
    if retry <= 0:
        return None
    try:
        params = {"uuid": uuid}
        res = requests.get(api_url, params=params, timeout=30)
        return res.json()
    except requests.RequestException as e:
        print("RequestException: %s" % str(e))
        return request_sku_info(uuid, retry - 1)


def get_uuid_or_pinyin(uuid):
    if not FLAGS.cvt_uuid_to_pinyin:
        return uuid
    info = request_sku_info(uuid)
    pinyin = info.get("data", {}).get("name", {}).get("pinyin")
    if not pinyin:
        print("Cannot get pinyin of %s" % uuid)
        pinyin = uuid
    return pinyin


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
        if child.tag not in ('object', 'point'):
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


def main(_):
    image_names = [n for n in os.listdir(FLAGS.images_dir)
                   if n.lower().rsplit(".")[-1] in ("jpg", "jpeg", "png") and not n.startswith(".")]
    image_count = len(image_names)
    print("Found %s images in %s" % (image_count, FLAGS.images_dir))

    label_map_dict = {}
    category_index = {}
    uuid2pinyin_map = {}
    prs = tqdm.tqdm(total=len(image_names))
    for idx, img_name in enumerate(image_names):
        name_head = img_name.rsplit(".", 1)[0]
        img_path = os.path.join(FLAGS.images_dir, img_name)
        annotation_path = os.path.join(FLAGS.annotations_dir, name_head + ".xml")

        img = Image.open(img_path).convert("RGB")
        img_arr = np.array(img)
        annotation = _read_annotation(annotation_path)

        xmin, xmax, ymin, ymax = [], [], [], []
        classes = []
        for obj in annotation["object"]:
            if "bndbox" not in obj:
                continue
            xmin.append(float(obj["bndbox"]["xmin"]))
            xmax.append(float(obj["bndbox"]["xmax"]))
            ymin.append(float(obj["bndbox"]["ymin"]))
            ymax.append(float(obj["bndbox"]["ymax"]))
            if obj["name"] not in uuid2pinyin_map:
                uuid2pinyin_map[obj["name"]] = get_uuid_or_pinyin(obj["name"])
            obj["name"] = uuid2pinyin_map[obj["name"]]
            if obj["name"] not in label_map_dict:
                label_map_dict[obj["name"]] = len(label_map_dict) + 1
                t = {"name": obj["name"], "id": label_map_dict[obj["name"]]}
                category_index[label_map_dict[obj["name"]]] = t
            classes.append(label_map_dict[obj["name"]])
        boxes = np.asarray([ymin, xmin, ymax, xmax]).transpose([1, 0])
        classes = np.asarray(classes, dtype=np.int32)
        scores = np.ones_like(ymin) * 0.99
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            img_arr, boxes, classes=classes,
            scores=scores, category_index=category_index,
            use_normalized_coordinates=False,
            line_thickness=4,
            max_boxes_to_draw=None,
            skip_scores=True)

        polygons = []
        classes = []
        for obj in annotation["object"]:
            if "polygon" not in obj:
                continue
            # plg = [[int(pt['x']), int(pt['y'])] for pt in obj["polygon"]["point"]]
            plg = [[float(pt['x']), float(pt['y'])] for pt in obj["polygon"]["point"]]
            polygons.append(np.asarray(plg))
            if obj["name"] not in uuid2pinyin_map:
                uuid2pinyin_map[obj["name"]] = get_uuid_or_pinyin(obj["name"])
            obj["name"] = uuid2pinyin_map[obj["name"]]
            if obj["name"] not in label_map_dict:
                label_map_dict[obj["name"]] = len(label_map_dict) + 1
                t = {"name": obj["name"], "id": label_map_dict[obj["name"]]}
                category_index[label_map_dict[obj["name"]]] = t
            classes.append(label_map_dict[obj["name"]])
        classes = np.asarray(classes, dtype=np.int32)
        scores = np.ones_like(classes) * 0.99
        visualization_utils.visualize_polygons_and_labels_on_image_array(
            img_arr, polygons, classes=classes,
            scores=scores, category_index=category_index,
            use_normalized_coordinates=False,
            line_thickness=4,
            max_polygons_to_draw=None,
            skip_scores=True)

        if not os.path.exists(FLAGS.outputs_dir):
            os.mkdir(FLAGS.outputs_dir)
        img = Image.fromarray(img_arr)
        img.save(os.path.join(FLAGS.outputs_dir, img_name))

        prs.update(1)


if __name__ == "__main__":
    tf.app.run()
