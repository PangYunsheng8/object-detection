# encoding=utf-8

"""
This script will purify annotation, remove some invalid boxes and
specified classes and repeated boxes.
"""

import os
import glob
import json
from lxml import etree
import argparse
import re

import tqdm
import numpy as np


def _recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` and `point` tags are the only twos that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if xml is None:
        return {}
    if len(xml) == 0:
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


def _recursive_create_dict_to_xml(dct, root):
    """Recursively create XML contents base on a python dict.

    Args:
      dct: python dictionary holding XML contents
      root: xml tree root where dict contents will append on.

    Returns:
    """
    for key, val in dct.items():
        if isinstance(val, dict):
            node = etree.SubElement(root, key)
            _recursive_create_dict_to_xml(val, node)
        elif isinstance(val, list):
            for x in val:
                node = etree.SubElement(root, key)
                _recursive_create_dict_to_xml(x, node)
        else:
            node = etree.SubElement(root, key)
            node.text = str(val)


def _read_annotation(annotation_path):
    with open(annotation_path, 'rb') as f:
        xml_str = f.read()
    xml = etree.fromstring(xml_str)
    data = _recursive_parse_xml_to_dict(xml)["annotation"]
    return data


def _write_annotation(annotation_path, annotation):
    root = etree.Element("annotation")
    _recursive_create_dict_to_xml(annotation, root)  # write to file:
    tree = etree.ElementTree(root)
    tree.write(annotation_path, pretty_print=True, encoding='utf-8')


def main(_):
    assert FLAG.annotation_folder != ""

    exclude_classes = []
    if FLAG.exclude_classes:
        exclude_classes = FLAG.exclude_classes

    if FLAG.purify_annotation_folder == "":
        FLAG.purify_annotation_folder = FLAG.annotation_folder
    if not os.path.exists(FLAG.purify_annotation_folder):
        os.makedirs(FLAG.purify_annotation_folder)

    FLAG.annotation_folder = FLAG.annotation_folder.rstrip("/") + "/"
    FLAG.purify_annotation_folder = FLAG.purify_annotation_folder.rstrip("/") + "/"

    if FLAG.recursive:
        ann_paths = glob.glob(os.path.join(FLAG.annotation_folder, "**", "*.xml"), recursive=True)
    else:
        ann_paths = glob.glob(os.path.join(FLAG.annotation_folder, "*.xml"))
    for path in tqdm.tqdm(ann_paths):
        annotation = _read_annotation(path)
        # import pdb
        # pdb.set_trace()
        objs = []
        bndboxes = {}
        for obj in annotation.get("object", []):
            if "bndbox" not in obj:
                if "polygon" in obj:
                    pts = [(int(round(float(pt['x']))), int(round(float(pt['y'])))) for pt in obj["polygon"]["point"]]
                    pts = np.asarray(pts)
                    xmin, ymin = np.min(pts, axis=0)
                    xmax, ymax = np.max(pts, axis=0)
                    obj["bndbox"] = {"xmin": xmin, "ymin": ymin,
                                     "xmax": xmax, "ymax": ymax}
                else:
                    print("Warning: no bndbox or polygon in object")
                    continue
            xmin = obj["bndbox"]["xmin"]
            ymin = obj["bndbox"]["ymin"]
            xmax = obj["bndbox"]["xmax"]
            ymax = obj["bndbox"]["ymax"]
            if int(xmax) - int(xmin) <= 0 or int(ymax) - int(ymin) <= 0:
                print("Invalid box: \n" + json.dumps(obj, indent=4, ensure_ascii=False))
                continue
            if obj["name"].strip() in exclude_classes:
                print("Invalid class name: \n" + json.dumps(obj, indent=4, ensure_ascii=False))
                continue
            w, h = int(xmax) - int(xmin), int(ymax) - int(ymin)
            if min(w, h) < FLAG.min_short_edge:
                print("Short edge is less than %s \n" % FLAG.min_short_edge
                      + json.dumps(obj, indent=4, ensure_ascii=False))
                continue
            if max(w, h) < FLAG.min_long_edge:
                print("Long edge is less than %s \n" % FLAG.min_long_edge
                      + json.dumps(obj, indent=4, ensure_ascii=False))
                continue
            bndbox_str = "%s_%s_%s_%s" % (xmin, ymin, xmax, ymax)
            if bndbox_str not in bndboxes:
                bndboxes[bndbox_str] = obj["name"]
            else:
                print("Repeated bndbox: %s %s" % (obj["name"], bndbox_str))
                if bndboxes[bndbox_str] != obj["name"]:
                    print("\033[1;31mWarning: Repeated bndboxes have different class name (%s <> %s)! \033[0m" % (
                        bndboxes[bndbox_str], obj["name"]))
                continue
            objs.append(obj)
        annotation["object"] = objs
        res_path = re.sub("^"+FLAG.annotation_folder, FLAG.purify_annotation_folder, path)
        d = os.path.dirname(res_path)
        if not os.path.exists(d):
            os.makedirs(d)
        _write_annotation(res_path, annotation)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_folder", required=True,
                        help="The annotation directory")
    parser.add_argument("--exclude_classes", nargs='*',
                        help="The exclude class names")
    parser.add_argument("--purify_annotation_folder", required=True,
                        help="The result annotation directory.")
    parser.add_argument("--min_short_edge", type=int, default=0,
                        help="the minimum value of short edge.")
    parser.add_argument("--min_long_edge", type=int, default=0,
                        help="the minimum value of long edge.")
    parser.add_argument("--recursive", action="store_true", default=False,
                        help="recursive purify annotations files in annotation_folder.")
    return parser.parse_args()


if __name__ == '__main__':
    FLAG = get_args()
    main("")
