# encoding=utf-8

import os
import sys
import random
import hashlib
import json

import tqdm
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

from lxml import etree

FLAG = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("front_dir", "",
                           "Front images directory, image should be png format")
tf.app.flags.DEFINE_string("background_dir", "",
                           "Front images directory")
tf.app.flags.DEFINE_string("output_dir", "",
                           "Output directory")
tf.app.flags.DEFINE_integer("count", 1000,
                            "The number of example image to generate. default: 1000")


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


def write_instance_masks(instance_mask_path, instance_masks):
    with open(instance_mask_path, "w") as f:
        out = json.dumps(instance_masks, indent=4)
        f.write(out)


def get_all_background_images(background_dir):
    images = []
    for name in os.listdir(background_dir):
        try:
            path = os.path.join(background_dir, name)
            im = cv2.imread(path)
            height, width = im.shape[:2]
            if height >= 512 and width >= 384:
                images.append(im)
        except:
            pass
    return images


def get_all_front_images(front_dir):
    front_images = {}
    for sub_folder in sorted(os.listdir(front_dir)):
        sub_folder_ = os.path.join(front_dir, sub_folder)
        for name in sorted(os.listdir(sub_folder_)):
            if sub_folder not in front_images:
                front_images[sub_folder] = []
            front_img_path = os.path.join(sub_folder_, name)
            im = cv2.imread(front_img_path, cv2.IMREAD_UNCHANGED)
            front_images[sub_folder].append(im)
    return front_images


def paste(back_images, front_images, sku_scale_map, output_dir, with_mask=False):
    front_classes = list(front_images.keys())

    # layout scheme
    min_scale, max_scale = (0.08, 0.18)
    rnd_scale = random.random() * (max_scale - min_scale) + min_scale
    rnd_back_img = back_images[random.randint(0, len(back_images) - 1)]
    b_height, b_width = rnd_back_img.shape[:2]
    unit_height = int(round(b_height * rnd_scale))
    unit_height = max(48, unit_height)
    shelf_height = int(round(unit_height * 1.9))
    shelf_number = (b_height - 100 - unit_height) // shelf_height
    shelves = []
    for i in range(shelf_number):
        shelves.append(
            [50, 50 + i * shelf_height, b_width - 50, 50 + (i + 1) * shelf_height])
    max_off_x = int(round(random.random() * 0.5 * shelf_height))
    max_off_y = int(round(random.random() * 0.1 * shelf_height))

    gen_img = np.copy(rnd_back_img)
    boxes = []
    classes = []
    instance_masks = []
    for shelf in reversed(shelves):
        xmin, ymin, xmax, ymax = shelf
        while True:
            rnd_class = front_classes[random.randint(0, len(front_classes) - 1)]
            rnd_class_images = front_images[rnd_class]
            rnd_idx = random.randint(0, len(rnd_class_images) - 1)
            item_img = rnd_class_images[rnd_idx]
            height, width = item_img.shape[:2]
            new_height = int(round(unit_height * sku_scale_map[rnd_class]))
            new_width = int(round(width * new_height / height))
            item_img = np.asarray(Image.fromarray(item_img).resize((new_width, new_height), Image.BILINEAR))
            height, width = item_img.shape[:2]
            off_x = random.randint(0, max_off_x)
            off_y = random.randint(-max_off_y, max_off_y)
            r_xmin = xmin + off_x
            r_xmax = xmin + off_x + width
            r_ymin = ymax - off_y - height
            r_ymax = ymax - off_y
            if r_xmax > xmax or r_ymin < 0:
                break
            if item_img.shape[-1] == 4:
                alpha = item_img[:, :, -1:] / 255.
                item_img = (1 - alpha) * gen_img[r_ymin:r_ymax, r_xmin:r_xmax, :] + \
                           alpha * item_img[:, :, :-1]
                item_img = item_img.astype(np.uint8)
            gen_img[r_ymin:r_ymax, r_xmin:r_xmax, :] = item_img
            boxes.append([r_xmin, r_ymin, r_xmax, r_ymax])
            classes.append(rnd_class)
            if with_mask:
                assert front_images[rnd_class][rnd_idx].shape[-1] == 4, \
                    "Front image must have alpha channel" + str((rnd_class, rnd_idx))
                instance_masks.append((rnd_class, rnd_idx))
            xmin = r_xmax

    if len(boxes) == len(classes) > 0:
        name_head = str(hash(gen_img.tostring()))

        if not os.path.exists(os.path.join(output_dir, "Images")):
            tf.gfile.MakeDirs(os.path.join(output_dir, "Images"))
        img_path = os.path.join(output_dir, "Images", name_head + ".jpg")

        if not os.path.exists(os.path.join(output_dir, "Annotations")):
            tf.gfile.MakeDirs(os.path.join(output_dir, "Annotations"))
        annotation_path = os.path.join(output_dir, "Annotations", name_head + ".xml")

        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        gen_img = Image.fromarray(gen_img)
        width, height = gen_img.size
        gen_img.save(img_path)
        write_detection_annotation(annotation_path, img_path, width, height, boxes, classes)
        if with_mask:
            if not os.path.exists(os.path.join(output_dir, "Mask")):
                tf.gfile.MakeDirs(os.path.join(output_dir, "Mask"))
            ins_mask_path = os.path.join(output_dir, "Mask", name_head + ".mask")
            write_instance_masks(ins_mask_path, instance_masks)
        return 1
    else:
        return 0


def main(_):
    back_images = get_all_background_images(FLAG.background_dir)
    front_images = get_all_front_images(FLAG.front_dir)

    if len(back_images) <= 0:
        print("Must have background images.")
        sys.exit(1)
    if len(front_images) <= 0:
        print("Must have front images.")
        sys.exit(1)

    sku_scale_map = {
        '330ml Cola': 1.0,
        '330ml Fanta': 1.0,
        '330ml Light Cola': 1.13,
        '330ml Zero Cola': 1.15,
        '500ml Zero Cola': 1.60,
        '600ml Cola': 1.85,
        '600ml Fanta': 1.85,
        '600ml Sprite': 1.67,
        '2L Cola': 2.10,
        '2L Sprite': 2.10,
    }

    process_bar = tqdm.tqdm(total=FLAG.count)
    crt = 0
    while crt < FLAG.count:
        res = paste(back_images, front_images, sku_scale_map, FLAG.output_dir, with_mask=True)
        crt += res
        process_bar.update(res)
    process_bar.close()


if __name__ == '__main__':
    tf.app.run()
