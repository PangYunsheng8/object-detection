# encoding=utf-8

import os

from lxml import etree
import tqdm
from PIL import Image
import numpy as np
import tensorflow as tf

from det_cls_api import ColaDetector, ColaClassifier, ColaDetectorClassify

import sys

prj_fld = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(prj_fld, "detection", "object_detection", "utils")
print(path)
if path not in sys.path:
    sys.path.append(path)
import visualization_utils

FLAG = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("mod", 'detection',
                           'Type of process. Can be '
                           'one of [`detection`, `classification`, `detection_classification`].'
                           'Default is `detection`')
tf.app.flags.DEFINE_string("model_path", "",
                           "Path of frozen model")
tf.app.flags.DEFINE_string("image_path_or_dir", "",
                           "Path or directory of images")
tf.app.flags.DEFINE_string("output_dir", "",
                           "The directory of process outputs")
tf.app.flags.DEFINE_boolean("recursive", False,
                            "if image_path_or_dir is a directory, collect all images "
                            "contained in the directory recursively")


def collect_images(image_folder, recursive=False):
    image_paths = []
    for name in os.listdir(image_folder):
        path = os.path.join(image_folder, name)
        if recursive is True and os.path.isdir(path):
            ps = collect_images(path, recursive=recursive)
            image_paths += ps
        elif name.lower().rsplit(".", 1)[-1] in ("jpg", "jpeg", "png"):
            image_paths.append(path)
    return image_paths


def write_detection_annotation(annotation_path, image_path, image_width, image_height, boxes, class_names):
    assert len(boxes) == len(class_names)

    root = etree.Element("annotation")

    file_name = etree.SubElement(root, "filename")
    file_name.text = str(os.path.split(image_path)[-1])

    path = etree.SubElement(root, "path")
    path.text = str(image_path)

    size = etree.SubElement(root, "size")
    width = etree.SubElement(size, "width")
    width.text = str(image_width)
    height = etree.SubElement(size, "height")
    height.text = str(image_height)
    depth = etree.SubElement(size, "depth")
    depth.text = str(3)

    for (xmin, ymin, xmax, ymax), class_name in zip(boxes, class_names):
        obj = etree.SubElement(root, "object")
        name = etree.SubElement(obj, "name")
        name.text = str(class_name)
        pose = etree.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = etree.SubElement(obj, "truncated")
        truncated.text = str(0)
        difficult = etree.SubElement(obj, "difficult")
        difficult.text = str(0)
        bndbox = etree.SubElement(obj, "bndbox")
        child11 = etree.SubElement(bndbox, "xmin")
        child11.text = str(xmin)
        child12 = etree.SubElement(bndbox, "ymin")
        child12.text = str(ymin)
        child13 = etree.SubElement(bndbox, "xmax")
        child13.text = str(xmax)
        child14 = etree.SubElement(bndbox, "ymax")
        child14.text = str(ymax)

    # write to file:
    tree = etree.ElementTree(root)
    tree.write(annotation_path, pretty_print=True, encoding='utf-8')


def main(_):
    image_paths = [FLAG.image_path_or_dir]
    if os.path.isdir(FLAG.image_path_or_dir):
        image_paths = collect_images(FLAG.image_path_or_dir, FLAG.recursive)

    if not tf.gfile.IsDirectory(FLAG.output_dir):
        tf.gfile.MakeDirs(FLAG.output_dir)

    if FLAG.mod == "detection":
        detector = ColaDetector(FLAG.model_path)
        category_index = {}
        for k, v in detector.index_label_map.items():
            if "display_name" in v:
                category_index[v['id']] = {'id': v['id'], 'name': v['display_name']}
            else:
                category_index[v['id']] = {'id': v['id'], 'name': v['class']}

        for img_path in tqdm.tqdm(image_paths):
            image = Image.open(img_path)
            boxes, scores, classes, num = detector.get_detection_result(image, threshold=0.5)
            # visualize boxes and labels on image
            ann_img_path = os.path.join(FLAG.output_dir, os.path.split(img_path)[-1])
            image_np = np.array(image)
            visualization_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes, axis=0),
                np.squeeze(classes, axis=0).astype(np.int32),
                np.squeeze(scores, axis=0),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=0.5,
                max_boxes_to_draw=None)
            img = Image.fromarray(image_np)
            img.save(ann_img_path)
            # write detection result as detection annotation format
            boxes, scores, classes = boxes[0], scores[0], classes[0]
            width, height = image.size
            ird = lambda x: int(round(x))
            # xmin, ymin, xmax, ymax
            image_coordinate_boxes = [[ird(x[1] * width), ird(x[0] * height), ird(x[3] * width), ird(x[2] * height)]
                                      for x in boxes]
            class_names = [detector.index_label_map[x] for x in classes]
            name_head = os.path.split(img_path)[-1].rsplit(".", 1)[0]
            ann_path = os.path.join(FLAG.output_dir, name_head + ".xml")
            write_detection_annotation(annotation_path=ann_path,
                                       image_path=img_path,
                                       image_width=width,
                                       image_height=height,
                                       boxes=image_coordinate_boxes,
                                       class_names=class_names)

    elif FLAG.mod == "classification":
        classifier = ColaClassifier(FLAG.model_path)

        for img_path in tqdm.tqdm(image_paths):
            image = Image.open(img_path)
            cls = classifier.get_top_1(image, return_label_name=True)
            output_path = os.path.join(FLAG.output_dir, cls)
            if not tf.gfile.IsDirectory(output_path):
                tf.gfile.MakeDirs(output_path)
            output_path = os.path.join(output_path, os.path.split(img_path)[-1])
            image.save(output_path)

    elif FLAG.mod == "detection_classification":
        detector_classifier = ColaDetectorClassify(FLAG.model_path)
        category_index = {}
        for k, v in detector_classifier.index_label_map.items():
            if "display_name" in v:
                category_index[v['id']] = {'id': v['id'], 'name': v['display_name']}
            else:
                category_index[v['id']] = {'id': v['id'], 'name': v['class']}

        for img_path in tqdm.tqdm(image_paths):
            image = Image.open(img_path)
            result = detector_classifier.detection_classify(image)
            boxes, scores, classes = result["boxes"], result["scores"], result["classes"]
            scores = np.expand_dims(scores, axis=0)
            classes = np.expand_dims(classes, axis=0)

            # visualize boxes and labels on image
            ann_img_path = os.path.join(FLAG.output_dir, os.path.split(img_path)[-1])
            image_np = np.array(image)
            visualization_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes, axis=0),
                np.squeeze(classes, axis=0).astype(np.int32),
                np.squeeze(scores, axis=0),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=0.0,
                max_boxes_to_draw=None)
            img = Image.fromarray(image_np)
            img.save(ann_img_path)
            # write detection result as detection annotation format
            boxes, scores, classes = boxes[0], scores[0], classes[0]
            width, height = image.size
            ird = lambda x: int(round(x))
            # xmin, ymin, xmax, ymax
            image_coordinate_boxes = [[ird(x[1] * width), ird(x[0] * height), ird(x[3] * width), ird(x[2] * height)]
                                      for x in boxes]
            class_names = [detector_classifier.index_label_map[x] for x in classes]
            name_head = os.path.split(img_path)[-1].rsplit(".", 1)[0]
            ann_path = os.path.join(FLAG.output_dir, name_head + ".xml")
            write_detection_annotation(annotation_path=ann_path,
                                       image_path=img_path,
                                       image_width=width,
                                       image_height=height,
                                       boxes=image_coordinate_boxes,
                                       class_names=class_names)

    else:
        raise Exception("Unexpected mod `%s`, please choose one of [`detection`, `classification`]")


if __name__ == '__main__':
    tf.app.run()
