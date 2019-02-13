# encoding=utf-8

import os
import sys
import json
from lxml import etree

import threadpool
import threading
from PIL import Image
from PIL import ImageFile
import tqdm
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

_RES = {}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", default="", help="Image folder")
    parser.add_argument("--annotation_folder", default="",
                        help="Annotation folder. This argument will be same as image_folder when it's not set.")
    parser.add_argument("--output_folder", default="", help="Output folder")
    parser.add_argument("--expand_ratio", type=float, default=0.,
                        help="Expand ratio of bounding box. Default is 0.")
    args = parser.parse_args()
    assert args.image_folder != "", "Must set image_folder"
    assert args.output_folder != "", "Must set output_folder"
    if args.annotation_folder == "":
        args.annotation_folder = args.image_folder
    return args


FLAG = get_args()


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


def get_patches(name):
    name_head = name.rsplit(".", 1)[0]
    img_path = os.path.join(FLAG.image_folder, name)
    ann_path = os.path.join(FLAG.annotation_folder, name_head + ".xml")

    image = Image.open(img_path)
    annotation = _read_annotation(ann_path)

    image_width, image_height = image.size

    for obj in annotation.get("object", []):
        xmin = int(obj["bndbox"]["xmin"])
        xmax = int(obj["bndbox"]["xmax"])
        ymin = int(obj["bndbox"]["ymin"])
        ymax = int(obj["bndbox"]["ymax"])

        h, w = ymax - ymin, xmax - xmin
        if h <= 0 or w <= 0:
            print("An invalid bounding box in '%s':" % ann_path)
            print(json.dumps(obj["bndbox"], indent=4))
            continue

        delta_h, delta_w = int(round(h * FLAG.expand_ratio / 2)), int(round(w * FLAG.expand_ratio / 2))
        ymi, yma = max(0, ymin - delta_h), min(image_height, ymax + delta_h)
        xmi, xma = max(0, xmin - delta_w), min(image_width, xmax + delta_w)

        sub_folder = os.path.join(FLAG.output_folder, obj["name"].strip())
        if not os.path.isdir(sub_folder):
            os.makedirs(sub_folder)
        ttt = "%s_%s_%s_%s.jpg" % (xmin, xmax, ymin, ymax)
        path = os.path.join(sub_folder, name + ttt)
        p_img = image.crop([xmi, ymi, xma, yma])
        p_img.save(path)

    if _RES["lock"].acquire():
        _RES["process"].update(1)
        _RES["lock"].release()


def main():
    assert FLAG.image_folder != "", "Must set image_folder"
    assert FLAG.output_folder != "", "Must set output_folder"

    if FLAG.annotation_folder == "":
        FLAG.annotation_folder = FLAG.image_folder

    image_names = os.listdir(FLAG.image_folder)
    image_names = [n for n in image_names if n.lower().rsplit(".", 1)[-1] in ("jpg", "png", "jpeg")]

    # create all sub-class folder
    class_set = set()
    ann_paths = [os.path.join(FLAG.annotation_folder, name.rsplit(".", 1)[0] + ".xml")
                 for name in image_names]
    for pt in ann_paths:
        annotation = _read_annotation(pt)
        for obj in annotation.get("object", []):
            if obj["name"] not in class_set:
                class_set.add(obj["name"])
                sub_folder = os.path.join(FLAG.output_folder, obj["name"].strip())
                if not os.path.isdir(sub_folder):
                    os.makedirs(sub_folder)
                    # print("Created class folder: ", sub_folder)
                    sys.stdout.write("\r" + " " * 80)
                    sys.stdout.write("\rCreated class folder: {fld}".format(fld=sub_folder))
                    sys.stdout.flush()

    print()
    pool = threadpool.ThreadPool(16)
    requests = threadpool.makeRequests(get_patches, image_names)
    _RES["process"] = tqdm.tqdm(requests)
    _RES["lock"] = threading.Lock()
    # map(pool.putRequest, requests)
    for req in requests:
        pool.putRequest(req)
    pool.wait()
    _RES["process"].refresh()


if __name__ == '__main__':
    main()
