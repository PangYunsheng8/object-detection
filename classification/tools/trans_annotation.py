# encoding=utf-8

import os
import json
import re

from lxml import etree


def _recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
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
        if child.tag != 'rect':
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
    data = _recursive_parse_xml_to_dict(xml)["root"]
    return data


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


def test():
    folder = "/home/zhaojh/Datasets/冰箱标注"
    xml_names = [n for n in os.listdir(folder) if n.endswith(".xml")]
    count = 0
    for name in xml_names:
        path = os.path.join(folder, name)
        data = _read_annotation(path)
        rect_list = data["modifyhistory"]["modifyitem"]["rectlist"]["rect"]
        count += len(rect_list)
        pass
    print(count)


def trans():
    folder = "/home/zhaojh/Datasets/冰箱标注"
    output_folder = "/home/zhaojh/Datasets/冰箱标注_annotation"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    drink_name_map = {
        "xuebi": "Sprite",
        "Schweppes": "Schweppes+C",
        "other": "qita"
    }
    not_include = [
        "binghongcha",
        "bingtangxueli",
        "fengmiyouzi",
        "guozhi",
        "haijingningmeng",
        "jingdiannaicha",
        "lvcha",
        "molimicha",
        "moliqingcha",
        "suanmeitang",
        "yinyongshui"
    ]

    xml_names = [n for n in os.listdir(folder) if n.endswith(".xml")]
    t = []
    files = []
    for name in xml_names:
        path = os.path.join(folder, name)
        data = _read_annotation(path)

        ann_path = os.path.join(output_folder, name)
        image_path = "xxx.jpg"
        image_width = data["size"]["width"]
        image_height = data["size"]["height"]
        boxes, classes = [], []
        rect_list = data["modifyhistory"]["modifyitem"]["rectlist"]["rect"]
        for rect in rect_list:
            if rect["drinkname"] in drink_name_map:
                rect["drinkname"] = drink_name_map.get(rect["drinkname"])
            class_name = " ".join([rect.get(att) for att in ["drinkname", "guige", "kouwei"] if rect.get(att)])
            if rect["drinkname"] in not_include:
                class_name = "qita"
            # if len(class_name) <= 0:
            #     print(json.dumps(rect, indent=2, ensure_ascii=False))
            x, y, w, h = [int(rect.get(att)) for att in ["x", "y", "width", "height"]]
            if " " in class_name and (re.findall("\d+ml|[\d\.]+L", class_name)):
                classes.append(class_name)
                boxes.append([x, y, x + w, y + h])
            else:
                if rect["drinkname"] == "qita" or class_name == "qita":
                    class_name = "qita"
                    classes.append(class_name)
                    boxes.append([x, y, x + w, y + h])
                else:
                    if len(files) <= 0 or files[-1] != name:
                        print(name)
                    print(json.dumps(rect, indent=2, ensure_ascii=False))
                    files.append(name)
        t += classes
        write_detection_annotation(annotation_path=ann_path, image_path=image_path,
                                   image_width=image_width, image_height=image_height,
                                   boxes=boxes, class_names=classes)

    print(json.dumps(sorted(set(t)), indent=4))
    # print(len(set(t)))
    # print(json.dumps(sorted(set(files)), indent=4))


if __name__ == '__main__':
    test()
    trans()
