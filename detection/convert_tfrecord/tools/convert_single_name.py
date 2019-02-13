# -*- coding: utf-8 -*-
# @Author: Yongqiang Qin
# @Date:   2018-07-12 23:04:35
# @Last Modified by:   Yongqiang Qin
# @Last Modified time: 2018-07-17 20:10:08
import xml.etree.ElementTree as ET
import pickle
import os
import argparse
from voc_ann import GEN_Annotations

def convert_with_folder_structure(input_root_dir, output_root_dir, new_class_name):

    input_root_dir = os.path.normpath(os.path.join(input_root_dir, ''))
    output_root_dir = os.path.normpath(os.path.join(output_root_dir, ''))

    for root, dirs, files in os.walk(input_root_dir):
        input_dir = root
        output_dir = root.replace(input_root_dir, output_root_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        read_and_convert(input_dir, output_dir, new_class_name)

def read_and_convert(ori_xml_dir, res_xml_dir, new_class_name):
    files = os.listdir(ori_xml_dir)
    xml_files = [_ for _ in files if _.endswith('xml')]

    for x in xml_files:
        new_path = os.path.join(res_xml_dir, x)
        ori_path = os.path.join(ori_xml_dir, x)
        tree = ET.parse(ori_path)
        root = tree.getroot()
        
        new_xml = GEN_Annotations(folder = res_xml_dir,
                                    filename = x,
                                    path = new_path)
        size = root.find('size')
        if size is not None:
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            c = int(size.find('channel').text)
            new_xml.set_size(w, h, c)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            # cls_name = 'Unspecify_Class_Name'
            
            xmlbox = obj.find('bndbox')
            new_xml.add_pic_attr_xyxy(new_class_name,
                                    xmlbox.find('xmin').text, 
                                    xmlbox.find('ymin').text, 
                                    xmlbox.find('xmax').text, 
                                    xmlbox.find('ymax').text)

        new_xml.savefile(new_path)


if __name__ == '__main__':
    # ori_xml_dir = 'xml'
    # res_xml_dir = 'xml_1_name'

    # if not os.path.exists(res_xml_dir):
    #   os.makedirs(res_xml_dir)

    # assert(ori_xml_dir != res_xml_dir)

    # read_and_convert(ori_xml_dir, res_xml_dir)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        type=str,
                        required=True,)
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,)
    parser.add_argument('--class_name',
                        default='Unspecify_Class_Name',
                        type=str)
    args = parser.parse_args()

    input_root_dir = args.input_dir
    output_root_dir = args.output_dir

    assert(input_root_dir != output_root_dir)

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    convert_with_folder_structure(input_root_dir, output_root_dir, args.class_name)

