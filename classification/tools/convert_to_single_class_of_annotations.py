#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-27 下午5:15
# @Author  : Jh Zhao
# @Site    :
# @File    : convert_single_name_of_annotations.py.py
# @Software: PyCharm Community Edition

import os
import glob
import argparse
from lxml import etree
import multiprocessing
import json
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


def convert_single_name_one_process(prs_id, src_dst_paths):
    for src_pth, dst_pth in tqdm.tqdm(src_dst_paths, desc="prs-%02d" % prs_id, position=prs_id):
        ann = _read_annotation(src_pth)
        objs = []
        for obj in ann.get("object", []):
            obj["name"] = new_class_name
            objs.append(obj)
        ann['object'] = objs
        dst_dir = os.path.dirname(dst_pth)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        _write_annotation(dst_pth, ann)


def list_split(lst, num):
    import math
    n = int(math.ceil(len(lst) / num))
    for i in range(num):
        yield lst[i * n:(i + 1) * n]


def convert_single_name(src_dst_paths, prs_num):
    prs_lst = []
    for i, p in enumerate(list_split(src_dst_paths, prs_num)):
        p = multiprocessing.Process(target=convert_single_name_one_process, args=(i, p))
        p.start()
        prs_lst.append(p)
    for p in prs_lst:
        p.join()


def load_json(p):
    with open(p) as f:
        return json.load(f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="annotation directory")
    parser.add_argument("--output_dir", required=True,
                        help="result directory")
    parser.add_argument("--class_name", required=True,
                        help="new class new")
    parser.add_argument("--prs_num", type=int, default=16,
                        help="merge process number.")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    new_class_name = args.class_name
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ann_paths = glob.glob(os.path.join(args.input_dir, '**', '*.xml'), recursive=True)
    ann_paths += glob.glob(os.path.join(args.input_dir, '**', '*.XML'), recursive=True)
    dst_ann_paths = [x.replace(args.input_dir.rstrip('/'), args.output_dir.rstrip('/'), 1) for x in ann_paths]
    pairs = list(zip(ann_paths, dst_ann_paths))
    convert_single_name(pairs, args.prs_num)
