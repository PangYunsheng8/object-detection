#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-28 下午3:41
# @Author  : Jh Zhao
# @Site    : 
# @File    : collect_detection_dataset.py
# @Software: PyCharm Community Edition

import os
import argparse
import json
import logging
import tqdm


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
    annotations = {}
    for d in tqdm.tqdm(dirs, desc="collect annotations"):
        names = [x for x in os.listdir(d) if x.rsplit(".", 1)[-1].lower() == 'xml']
        heads = [get_name_head(x) for x in names]
        paths = [os.path.join(d, x) for x in names]
        annotations.update(dict(zip(heads, paths)))
    return annotations


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dirs", nargs="+",
                        help="image directories")
    parser.add_argument("--annotations_dirs", nargs="+",
                        help="annotation directories")
    parser.add_argument("--output_path", required=True,
                        help="the output json file path")
    _args = parser.parse_args()
    assert len(_args.image_dirs) == len(_args.annotations_dirs), "image_dirs and annotations_dirs must match"
    return _args


if __name__ == '__main__':
    args = get_args()

    images = collect_images(args.image_dirs)
    annotations = collect_annotations(args.annotations_dirs)

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
    inter = list(inter)
    img_paths = [images[x] for x in inter]
    ann_paths = [annotations[x] for x in inter]
    ds = {"image_paths": img_paths, "annotation_paths": ann_paths}
    with open(args.output_path, 'w') as f:
        json.dump(ds, f, indent=2)
