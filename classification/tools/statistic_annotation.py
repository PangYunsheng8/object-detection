#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-10 下午4:15
# @Author  : Jh Zhao
# @Site    : 
# @File    : statistic_annotation.py
# @Software: PyCharm Community Edition

import os
import glob
import argparse
from lxml import etree
from collections import Counter
import matplotlib.pyplot as plt
# import pandas as pd
import tqdm
import numpy as np


def _recursive_parse_xml_to_dict(xml):
    if xml is None:
        return {}
    if len(xml) == 0:
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


def statistic_annotations(ann_dir, depth):
    flt = os.path.join(ann_dir, *("*" * depth)) + ".xml"
    ann_paths = glob.glob(flt)
    objs = []
    for p in tqdm.tqdm(ann_paths, desc="scanning"):
        ann = _read_annotation(p)
        for obj in ann.get("object", []):
            objs.append(obj["name"])
    count = Counter(objs)
    
    y = np.arange(len(count))
    t = count.most_common()
    y_tick, x = zip(*t)
    x = np.array(x)
    width = 0.8

    plt.figure(figsize=(16, 2 + len(count) // 2))
    plt.barh(y, x, width, log=True, tick_label=y_tick)
    plt.savefig("barh.png")
    plt.close()

    plt.figure(figsize=(16, 8))
    plt.hist(x, bins=1000)
    plt.savefig("hist.png")
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="annotation root folder")
    parser.add_argument("--depth", type=int, default=1, help="depth of xml file base on root folder")
    args_ = parser.parse_args()
    args_.folder = os.path.expanduser(os.path.expandvars(args_.folder))
    return args_


if __name__ == '__main__':
    args = get_args()
    statistic_annotations(args.folder, args.depth)
    # statistic_annotations(os.path.expanduser("~/Datasets/EBest_Cocacola/from_ebest/"), 1)
