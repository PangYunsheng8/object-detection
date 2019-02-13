# -*- coding: utf-8 -*-
# @Author: Yongqiang Qin
# @Date:   2018-07-20 09:31:16
# @Last Modified by:   Yongqiang Qin
# @Last Modified time: 2018-07-20 12:55:33
import os
import shutil
import argparse

def pair_with_folder_structure(src_root, dst_root, anno_root):
    src_root = os.path.normpath(os.path.join(src_root, ''))
    dst_root = os.path.normpath(os.path.join(dst_root, 'Images'))
    anno_root = os.path.normpath(os.path.join(anno_root, ''))

    for root, dirs, files in os.walk(src_root):
        src_image_dir = root
        dst_img_dir = root.replace(src_root, dst_root)
        anno_dir = root.replace(src_root, anno_root)
        if not os.path.exists(dst_img_dir):
            os.makedirs(dst_img_dir)

        copy_annotated_images(src_image_dir, dst_img_dir, anno_dir)
    

def copy_annotated_images(src_image_dir, dst_img_dir, anno_dir):
    assert(dst_img_dir != src_image_dir)
    img_files = os.listdir(src_image_dir)
    anno_files = os.listdir(anno_dir)
    img_files = [_ for _ in img_files if _.lower().endswith(('.jpg', '.png'))]
    anno_files = [_ for _ in anno_files if _.lower().endswith('.xml')]

    for img in img_files:
        anno_path = os.path.join(anno_dir, os.path.splitext(img)[0]+'.xml')
        if os.path.exists(anno_path):
            img_path = os.path.join(src_image_dir, img)
            dst_img_path = os.path.join(dst_img_dir, img)
            shutil.copy2(img_path, dst_img_path)

def copy_image_annotations(src_anno_dir, dst_anno_dir, img_dir):
    assert(src_anno_dir != dst_anno_dir)
    anno_files = os.listdir(src_anno_dir)
    img_files = os.listdir(img_dir)

    anno_files = [_ for _ in anno_files if _.lower().endswith('.xml')]
    img_files = [_ for _ in img_files if _.lower().endswith(('.jpg', '.png'))]

    for anno in anno_files:
        img_path = os.path.join(img_dir, os.path.splitext(anno)[0]+'.jpg')
        if os.path.exists(img_path):
            anno_path = os.path.join(src_anno_dir, anno)
            dst_anno_path = os.path.join(dst_anno_dir, anno)
            shutil.copy2(anno_path, dst_anno_path)

def copy_pair_files(opts):
    src_root = opts['src_root']
    dst_root = opts['dst_root']
    watcher_root = opts['watcher_root']

    src_ext = opts['src_ext']
    watcher_ext = opts['watcher_ext']

    src_root = os.path.normpath(os.path.join(src_root, ''))
    dst_root = os.path.normpath(os.path.join(dst_root, ''))
    watcher_root = os.path.normpath(os.path.join(watcher_root, ''))
    if not watcher_ext.startswith('.'):
        watcher_ext = '.' + watcher_ext


    for root, dirs, files in os.walk(src_root):
        src_files = [_ for _ in files if _.endswith(src_ext)]
        src_dir = root
        dst_dir = root.replace(src_root, dst_root)
        if os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        watcher_dir = root.replace(src_root, watcher_root)

        for f in src_files:
            watcher_file = os.path.join(watcher_dir, os.path.splitext(f)[0] + watcher_ext)
            if os.path.exists(watcher_file):
                src_file = os.path.join(src_dir, f)
                dst_file = os.path.join(dst_dir, f)
                shutil.copy2(src_file, dst_file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        type=str,
                        required=True,)
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,)
    parser.add_argument('--anno_dir',
                        type=str,
                        required=True,)
    args = parser.parse_args()

    src_root = args.input_dir
    dst_root = args.output_dir
    anno_root = args.anno_dir

    assert(src_root != dst_root)

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    pair_with_folder_structure(src_root, dst_root, anno_root)


