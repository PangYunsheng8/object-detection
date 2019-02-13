# encoding=utf-8

"""
This script will check the patches's class and it's annotation class.
The annotation class name will be rectified by patch's class folder name
if class folder name is not same as annotation class name.
"""

import os
import re
from lxml import etree
import glob

import tqdm
import tensorflow as tf

FLAG = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("patches_root", "",
                           "The root directory of all classes' patches")
tf.app.flags.DEFINE_string("annotation_folder", "",
                           "The annotation directory")
tf.app.flags.DEFINE_string("adjusted_annotation_folder", "",
                           "The adjusted result annotation directory. "
                           # "This will be same as annotation_folder if this augment is not set."
                           )


def _recursive_parse_xml_to_dict(xml):
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
    with open(annotation_path, 'rb') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = _recursive_parse_xml_to_dict(xml)["annotation"]
    return data


def _write_annotation(annotation_path, annotation):
    filename = annotation["filename"]
    image_path = annotation["path"]
    image_width = annotation["size"]["width"]
    image_height = annotation["size"]["height"]
    image_depth = annotation["size"]["depth"]

    root = etree.Element("annotation")

    file_name = etree.SubElement(root, "filename")
    file_name.text = str(filename)

    path = etree.SubElement(root, "path")
    path.text = str(image_path)

    size = etree.SubElement(root, "size")
    width = etree.SubElement(size, "width")
    width.text = str(image_width)
    height = etree.SubElement(size, "height")
    height.text = str(image_height)
    depth = etree.SubElement(size, "depth")
    depth.text = str(image_depth)

    for obj in annotation["object"]:
        obj_node = etree.SubElement(root, "object")
        name = etree.SubElement(obj_node, "name")
        name.text = str(obj["name"])
        pose = etree.SubElement(obj_node, "pose")
        pose.text = str(obj["pose"])
        truncated = etree.SubElement(obj_node, "truncated")
        truncated.text = str(obj["truncated"])
        difficult = etree.SubElement(obj_node, "difficult")
        difficult.text = str(obj["difficult"])
        bndbox = etree.SubElement(obj_node, "bndbox")
        child11 = etree.SubElement(bndbox, "xmin")
        child11.text = str(obj["bndbox"]["xmin"])
        child12 = etree.SubElement(bndbox, "ymin")
        child12.text = str(obj["bndbox"]["ymin"])
        child13 = etree.SubElement(bndbox, "xmax")
        child13.text = str(obj["bndbox"]["xmax"])
        child14 = etree.SubElement(bndbox, "ymax")
        child14.text = str(obj["bndbox"]["ymax"])

    # write to file:
    tree = etree.ElementTree(root)
    tree.write(annotation_path, pretty_print=True, encoding='utf-8')


def load_patches(patch_path):
    _reg_format = "\.[a-zA-Z]{3,4}(?=\d+)"
    _reg_patch_name = "^[\w]+(" + _reg_format + ")?\d+_\d+_\d+_\d+.jpg$"

    cls_names = os.listdir(patch_path)
    items = {}
    for cls in cls_names:
        cls_fld = os.path.join(patch_path, cls)
        img_names = os.listdir(cls_fld)
        for name in img_names:
            if not re.match(_reg_patch_name, name):
                continue
            name = re.sub(_reg_format, "", name)
            if name in items:
                print("Item [%s] is replacing class [%s] by [%s] !" % (name, items[name], cls))
            items[name] = cls
    return items


def load_annotations(annotation_folder):
    ann_paths = glob.glob(os.path.join(annotation_folder, "*.xml"))
    annotations = []
    items = {}
    for apt in ann_paths:
        ann = _read_annotation(apt)
        annotations.append(ann)
        image_name = os.path.split(apt)[-1].rsplit(".", 1)[0]
        for obj in ann.get("object", []):
            xmin = int(obj["bndbox"]["xmin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymin = int(obj["bndbox"]["ymin"])
            ymax = int(obj["bndbox"]["ymax"])
            if ymax - ymin <= 0 or xmax - xmin <= 0:
                continue
            bnd = (obj["bndbox"][x] for x in ["xmin", "xmax", "ymin", "ymax"])
            name = image_name + "_".join(bnd) + ".jpg"
            items[name] = obj["name"]
    return ann_paths, annotations, items


def main(_):
    assert FLAG.patches_root != ""
    assert FLAG.annotation_folder != ""

    assert FLAG.adjusted_annotation_folder != ""
    # if FLAG.adjusted_annotation_folder == "":
    #     FLAG.adjusted_annotation_folder = FLAG.annotation_folder
    if not tf.gfile.IsDirectory(FLAG.adjusted_annotation_folder):
        tf.gfile.MakeDirs(FLAG.adjusted_annotation_folder)

    patch_class_map = load_patches(FLAG.patches_root)

    ann_paths, annotations, ann_items = load_annotations(FLAG.annotation_folder)
    # check whether there are missing items
    ignore_or_remove = "i"
    miss_items = set(ann_items.keys()) - set(patch_class_map.keys())
    if len(miss_items) > 0:
        display_num = 10
        print("Missed [%s] items in patches root. Following are some patch ids:" % len(miss_items))
        print("\t" + "\n\t".join(list(miss_items)[:display_num]))
        if len(miss_items) > display_num:
            print("\t......")
        while True:
            ignore_or_remove = input("ignore or remove the missing items from annotation: ([i]/r) ")
            if ignore_or_remove == "":
                ignore_or_remove = "i"
            if ignore_or_remove in ("i", "r"):
                break
            else:
                print("please input i or r !")

    for apt, annotation in tqdm.tqdm(zip(ann_paths, annotations)):
        objs = []
        ann_name = os.path.split(apt)[-1]
        for obj in annotation["object"]:
            bnd = [obj["bndbox"][x] for x in ["xmin", "xmax", "ymin", "ymax"]]
            patch_name = ann_name[:-4] + "_".join(bnd) + ".jpg"
            if patch_name not in patch_class_map:
                if ignore_or_remove == "r":
                    continue
                else:
                    objs.append(obj)
            else:
                if patch_class_map.get(patch_name) != obj["name"]:
                    obj["name"] = patch_class_map.get(patch_name)
                objs.append(obj)
        annotation["object"] = objs
        res_path = os.path.join(FLAG.adjusted_annotation_folder, ann_name)
        _write_annotation(res_path, annotation)


if __name__ == '__main__':
    tf.app.run()
