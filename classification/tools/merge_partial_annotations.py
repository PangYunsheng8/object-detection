#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-19 下午5:40
# @Author  : Jh Zhao
# @Site    : 
# @File    : merge_dataset.py
# @Software: PyCharm Community Edition


import os
import argparse
import collections
import logging
import shutil
from lxml import etree
import tqdm


def _recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` and `point` tags are the only twos that can appear
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


def get_name_head(path):
    return os.path.basename(path).split(".", 1)[0]


def collect_images(dirs):
    images = {}
    for d in tqdm.tqdm(dirs, desc="collect images"):
        names = [x for x in os.listdir(d) if x.rsplit(".", 1)[-1].lower() in ('jpg', 'jpeg', 'png')]
        heads = [get_name_head(x) for x in names]
        paths = [os.path.join(d, x) for x in names]
        images.update(dict(zip(heads, paths)))
    return images


def collect_annotations(dirs):
    annotations = collections.defaultdict(list)
    for d in tqdm.tqdm(dirs, desc="collect annotations"):
        names = [x for x in os.listdir(d) if x.rsplit(".", 1)[-1].lower() == 'xml']
        for name in names:
            head = get_name_head(name)
            annotations[head].append(os.path.join(d, name))
    return annotations


def merge(image_dirs, annotation_dirs, output_dir):
    images = collect_images(image_dirs)
    annotations = collect_annotations(annotation_dirs)
    img_head_set = set(images.keys())
    ann_head_set = set(annotations.keys())
    no_ann = img_head_set - ann_head_set
    if no_ann:
        for x in list(no_ann)[:10]:
            logging.warning("with no annotation of %s" % x)
    no_img = ann_head_set - img_head_set
    if no_img:
        for x in list(no_img)[:10]:
            logging.warning("with no annotation of %s" % x)
    inter = img_head_set.intersection(ann_head_set)
    for head in tqdm.tqdm(inter, desc="merge annotation"):
        # move image
        dp = os.path.join(output_dir, os.path.basename(images[head]))
        shutil.copyfile(images[head], dp)
        # merge annotation
        anns = [_read_annotation(x) for x in annotations[head]]
        merged_ann = anns[0]
        for ann in anns[1:]:
            merged_ann['object'] += ann.get('object', [])
        dp = os.path.join(output_dir, head + ".xml")
        _write_annotation(dp, merged_ann)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dirs", nargs="+", required=True,
                        help="image directories")
    parser.add_argument("--annotation_dirs", nargs="+",
                        help="annotation directories, if do not set, it will be same as image_dirs")
    parser.add_argument("--output_dir", required=True,
                        help="merged dataset directory")
    _args = parser.parse_args()
    if not _args.annotation_dirs:
        _args.annotation_dirs = _args.image_dirs
    return _args


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    merge(args.image_dirs, args.annotation_dirs, args.output_dir)
