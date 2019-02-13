#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-19 上午11:35
# @Author  : Jh Zhao
# @Site    : 
# @File    : get_uuid2type_map.py
# @Software: PyCharm Community Edition

import requests
import json
import argparse

api_url = "http://47.92.9.46/label/name/"

short_name_of_type = {
    "瓶装": "pz",
    "箱装": "xz",
    "割箱": "gx",
    "袋装": "dz",
    "盒装": "hz",
    "杯装": "bz",
}


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


def load_label_index_map(label_index_map_path):
    with open(label_index_map_path) as f:
        label_index_map = json.load(f)
    if isinstance(label_index_map, list):
        label_index_map = {x['class']: x for x in label_index_map}
    else:
        label_index_map = {k: {"id": i, 'class': k} for k, i in label_index_map.items()}
    index_label_map = {v['id']: v for k, v in label_index_map.items()}
    return label_index_map, index_label_map


def get_uuid2type_map(label_index_map):
    result = {}
    for k in label_index_map:
        info = request_sku_info(k)
        type_name = info.get("data", {}).get("name", {}).get("type_name")
        if not type_name:
            print("Cannot get type name of %s" % k)
        else:
            result[k] = short_name_of_type[type_name]
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_index_map_path", required=True,
                        help="the path of label index map file")
    parser.add_argument("--result_path", required=True,
                        help="the path of result file")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    lim, _ = load_label_index_map(args.label_index_map_path)
    uuid2type_map = get_uuid2type_map(lim)
    with open(args.result_path, 'w') as f:
        json.dump(uuid2type_map, f, indent=2, ensure_ascii=False)
