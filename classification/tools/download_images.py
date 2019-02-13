#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-10 下午6:39
# @Author  : Jh Zhao
# @Site    :
# @File    : download_images.py
# @Software: PyCharm Community Edition

import os
import json
import argparse
import math
import time
import logging
from multiprocessing import Process

import threadpool
import requests
import tqdm

session = requests.session()
RES = {}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def download_one_image(url, tries=3):
    if tries <= 0:
        return None
    try:
        res = session.get(url, timeout=60)
        return res
    except:
        time.sleep(0.2)
        return download_one_image(url, tries=tries - 1)


def download_one_item(item):
    try:
        url = item['image_url']
        name = item['image_name']
        res = download_one_image(url)
        if res:
            try:
                output_path = os.path.join(RES['output_folder'], name)
                with open(output_path, "wb") as f:
                    f.write(res.content)
                item['image_name'] = output_path
                RES['success_list'].append(item)
            except Exception as e:
                logger.info("download failed! (%s)" % url)
                RES['fail_list'].append(item)
                logger.debug("error: %s, %s " % (e, json.dumps(item, ensure_ascii=False)))
        else:
            logger.info("download failed! (%s)" % url)
            RES['fail_list'].append(item)
    except Exception as e:
        logger.info("download failed!")
        RES['fail_list'].append(item)
        logger.debug("error: %s, %s " % (e, json.dumps(item, ensure_ascii=False)))
    RES['prs'].update()


def download_images(items, output_folder, prs_idx):
    RES['fail_list'] = []
    RES['success_list'] = []
    RES['output_folder'] = output_folder
    RES['prs'] = tqdm.tqdm(total=len(items), desc="prs%02d" % prs_idx, position=prs_idx)

    pool = threadpool.ThreadPool(16)
    reqs = threadpool.makeRequests(download_one_item, items)
    [pool.putRequest(req) for req in reqs]
    pool.wait()

    RES['prs'].refresh()

    info_folder = output_folder.rstrip("/") + "-info"
    with open(os.path.join(info_folder, "fail_%02d.json" % prs_idx), 'w') as f:
        json.dump(RES['fail_list'], f, indent=2)
    with open(os.path.join(info_folder, "success_%02d.json" % prs_idx), 'w') as f:
        json.dump(RES['success_list'], f, indent=2)
    logger.info("prs%02d download done!" % prs_idx)


def work(items, retry=3):
    for i in range(retry):
        part_num = math.ceil(len(items) / args.num)
        part_items = [items[i * part_num:(i + 1) * part_num] for i in range(args.num)]
        prs_lst = []
        for idx, pits in enumerate(part_items):
            prs = Process(target=download_images, args=(pits, args.output_dir, idx))
            prs_lst.append(prs)
            prs.start()
        for prs in prs_lst:
            prs.join()

        print("\n" * args.num)

        # merge all success and fail result
        names = os.listdir(dld_inf_fld)
        flt_lst = ["fail_%02d.json" % x for x in range(args.num) if "fail_%02d.json" % x in names]
        scs_lst = ["success_%02d.json" % x for x in range(args.num) if "success_%02d.json" % x in names]
        all_scs_lst = []
        for x in scs_lst:
            all_scs_lst += load_json(os.path.join(dld_inf_fld, x))
        all_flt_lst = []
        for x in flt_lst:
            all_flt_lst += load_json(os.path.join(dld_inf_fld, x))
        all_scs_file_path = os.path.join(dld_inf_fld, "all_success.json")
        if os.path.exists(all_scs_file_path):
            pre_all_scs = load_json(all_scs_file_path)
            assert isinstance(pre_all_scs, list)
            all_scs_lst = pre_all_scs + all_scs_lst
        save_json(all_scs_lst, all_scs_file_path)
        all_flt_file_path = os.path.join(dld_inf_fld, "all_fail.json")
        if len(all_flt_lst) != 0:
            items = all_flt_lst
            save_json(all_flt_lst, all_flt_file_path)
        else:
            break
    if len(all_flt_lst) == 0:
        logger.info("all download done!")
    else:
        logger.info("%s item(s) failed to download!" % len(all_flt_lst))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="json file")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("--num", type=int, default=16, help="number of process")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open(args.json_file) as f:
        dld_items = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dld_inf_fld = args.output_dir.rstrip("/") + "-info"
    if not os.path.exists(dld_inf_fld):
        os.makedirs(dld_inf_fld)

    # crate logger
    logger = logging.getLogger(os.path.basename(__file__))
    handler = logging.FileHandler(os.path.join(dld_inf_fld, "log.txt"))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    work(dld_items)

