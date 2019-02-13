# encoding=utf-8

import os
import json
from xml.dom.minidom import parse
import xml.dom.minidom
from collections import OrderedDict

import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import sys

crt_folder = os.path.dirname(__file__)


class Box:
    """
    BoundingBox info
    """

    def __init__(self, name, xmin, ymin, xmax, ymax):
        self.name = name  # drink name
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.match_box_id = -1  # current match box ID
        self.match_ratio = 0

    def __str__(self):
        return "name:{0} xmin:{1} ymin:{2} xmax:{3} ymax:{4}" \
            .format(self.name,
                    self.xmin, self.ymin,
                    self.xmax, self.ymax)

    def get_line_points(self):  # for draw Box
        return [(self.xmin, self.ymin), (self.xmax, self.ymin),
                (self.xmax, self.ymax), (self.xmin, self.ymax),
                (self.xmin, self.ymin)]


def load_json(path):
    """
    generate dictionary, convert test drink label to predict drink label(less types)
    :param path:Json file path and contains map info
    :return: dict structure
    """
    with open(path, 'r') as f:
        name_maps = json.load(f)
    return name_maps


def _phrase_xml(file, name_maps=None, is_test_image=False):
    """
    phrase XML info and get bounding boxes(list type)
    :param file: XML file path and contains boxes info
    :param name_maps:if TestIamge is True and name_maps is not
           the default Value=None,change the drink label(name)
    :param is_test_image:
    :return:list type,Box boxes
    """

    def swap(a, b):
        return b, a

    need_swap_x_y = False
    dom_tree = xml.dom.minidom.parse(file)
    annotation = dom_tree.documentElement
    objects = annotation.getElementsByTagName("object")
    boxes = []
    set_boxes = set()
    for obj in objects:
        name = obj.getElementsByTagName("name")[0].childNodes[0].data
        #name=name.encode('latin-1','ignore').decode('utf8',errors='ignore')
        bndbox = obj.getElementsByTagName("bndbox")[0]
        xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
        xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
        ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
        ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
        boxes_temp_value = (name, xmin, ymin, xmax, ymax)
        if boxes_temp_value in set_boxes:
            continue
        set_boxes.add(boxes_temp_value)
        if is_test_image and None != name_maps:  # If test Image and name_maps is not None ,\
            #  then replace it with the name of prediction name
            name = name_maps[name]
        else:
            if need_swap_x_y and (not is_test_image):  # prediction x,y error we need to change x,y
                xmin, ymin = swap(xmin, ymin)
                xmax, ymax = swap(xmax, ymax)
        boxes.append(Box(name, xmin, ymin, xmax, ymax))
    return boxes


def _calculate_box_match_ratio(box1, box2):
    """
    calculate box1 and box2 match value: overlap ratio
    :param box1: type class Box
    :param box2: type class Box
    :return: overlap ratio
    """
    xmin1, xmax1, ymin1, ymax1 = box1.xmin, box1.xmax, box1.ymin, box1.ymax
    xmin2, xmax2, ymin2, ymax2 = box2.xmin, box2.xmax, box2.ymin, box2.ymax
    overlap_x = (xmax1 - xmin1) + (xmax2 - xmin2) - (max(xmax1, xmax2) - min(xmin1, xmin2))
    overlap_y = (ymax1 - ymin1) + (ymax2 - ymin2) - (max(ymax1, ymax2) - min(ymin1, ymin2))
    if overlap_x <= 0 or overlap_y <= 0:
        return 0  # No overlap
    overlap_area = overlap_x * overlap_y
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    sum_area = area1 + area2 - overlap_area
    overlap_ratio = overlap_area * 1.0 / sum_area
    return overlap_ratio


def _compare_boxes(ground_truth_boxes, predict_boxes, iou_threshold=0.7):
    """
    compare boxes info get final result
    :param ground_truth_boxes: test_boxes got by _phrase_XML()
    :param predict_boxes: prediction_boxes got by _phrase_XML()
    :param iou_threshold: two boxes IOU ratio threshold ,default=0.7
    :return:boxes_test:modified boxes_test contains match info
            boxes_prediction：modified boxes_prediction contains match info
            match_type_error: match boxes but type different,list type (test_box,prediction_box)
            test_box_error: mismatch bndboxes which don't find
            predict_box_error: extra prediction bndbox
    """
    boxes_test_num = len(ground_truth_boxes)
    boxes_prediction_num = len(predict_boxes)
    match_list = []
    for i in range(boxes_test_num):
        for j in range(boxes_prediction_num):
            match_value = _calculate_box_match_ratio(ground_truth_boxes[i], predict_boxes[j])
            match_info = (i, j, match_value)
            match_list.append(match_info)
    match_list.sort(key=lambda x: float(x[2]), reverse=True)
    match_type_error = []  # match boxes but type different
    test_box_error = []  # mismatch bndboxes which don't find
    predict_box_error = []  # extra prediction bndbox
    match_num = 0
    for singleMatch in match_list:
        test_index, predict_index, match_value = singleMatch
        if match_value < iou_threshold:
            break
        if ground_truth_boxes[test_index].match_box_id != -1 \
                or predict_boxes[predict_index].match_box_id != -1:
            continue
        ground_truth_boxes[test_index].match_box_id = predict_index
        ground_truth_boxes[test_index].match_ratio = match_value
        predict_boxes[predict_index].match_box_id = test_index
        predict_boxes[predict_index].match_ratio = match_value
        match_num = match_num + 1
        if predict_boxes[predict_index].name != ground_truth_boxes[test_index].name:
            match_type_error.append((ground_truth_boxes[test_index], predict_boxes[predict_index]))
    for test_box in ground_truth_boxes:
        if test_box.match_box_id == -1:
            test_box_error.append(test_box)
    for predict_box in predict_boxes:
        if predict_box.match_box_id == -1:
            predict_box_error.append(predict_box)

    return ground_truth_boxes, predict_boxes, match_type_error, test_box_error, predict_box_error


def _draw_boxes(img_path, match_type_error, test_box_error, predict_box_error):
    im = Image.open(img_path)
    draw = ImageDraw.Draw(im)
    find_font=True
    try:
        my_font = ImageFont.truetype(os.path.join(crt_folder, 'Ubuntu-B.ttf'), size=15)
    except IOError:
        my_font = ImageFont.load_default()
        find_font=False
    for single_match_type_error in match_type_error:  # 匹配框类型不一致，黄颜色框，真实类别为红色字体，识别类别是蓝色
        test_box, predict_box = single_match_type_error
        draw.line(test_box.get_line_points(), fill='yellow', width=3)
        if not find_font:
            test_box.name =test_box.name.encode('latin-1', 'ignore').decode('utf8', errors='ignore')
            predict_box.name = predict_box.name.encode('latin-1', 'ignore').decode('utf8', errors='ignore')
        #print(test_box.name,predict_box.name)
        draw.text((test_box.xmin, test_box.ymin - 40), test_box.name, fill='red',font=my_font)
        draw.text((test_box.xmin, test_box.ymin - 20), predict_box.name,
                  fill='blue',font=my_font)
    for single_test_box_error in test_box_error:  # 漏框，真实的框没有被识别出来，红色框
        draw.line(single_test_box_error.get_line_points(), fill='red', width=3)
    for single_prediction_box_error in predict_box_error:  # 误报框，识别的框并不真实存在，蓝色框
        draw.line(single_prediction_box_error.get_line_points(), fill='blue', width=3)
    del draw
    return im


def deal_single_pair_xml(true_xml_file_path, predict_xml_file_path, iou_threshold=0.7, name_maps=None,
                         true_result=None, predict_result=None):
    """
   compare label's annotation and prediction's annotation and  return two dicts
   :param true_xml_file_path:  labeled annotation
   :param predict_xml_file_path:  predicted annotation
   :param name_maps: sku map brand dictionary if no need ,use the default None
   :param true_result:use the default
   :param predict_result:use the default
   :return: true_result(a dictionary contains label's annotation info)
           and  prediction_result(a dictionary contains prediction's annotation info)
    """
    if true_result is None:
        true_result = {}
    if predict_result is None:
        predict_result = {}

    boxes_test = _phrase_xml(true_xml_file_path, name_maps, True)
    boxes_prediction = _phrase_xml(predict_xml_file_path)
    boxes_test, boxes_prediction, match_type_error, test_box_error, predict_box_error \
        = _compare_boxes(boxes_test, boxes_prediction, iou_threshold=iou_threshold)
    for box in boxes_test:
        bok_name = box.name
        if bok_name not in true_result:
            true_result[bok_name] = {"correct_num": 0, "false_num": 0}
        if -1 == box.match_box_id:
            true_result[bok_name]["false_num"] += 1
        else:
            if bok_name != boxes_prediction[box.match_box_id].name:
                true_result[bok_name]["false_num"] += 1
            else:
                true_result[bok_name]["correct_num"] += 1
    for box in boxes_prediction:
        bok_name = box.name
        if bok_name not in predict_result:
            predict_result[bok_name] = {"correct_num": 0, "false_num": 0}
        if -1 == box.match_box_id:
            predict_result[bok_name]["false_num"] += 1
        else:
            if bok_name != boxes_test[box.match_box_id].name:
                predict_result[bok_name]["false_num"] += 1
            else:
                predict_result[bok_name]["correct_num"] += 1

    def sort_dict(dc):
        keys = []
        for key in dc:
            keys.append(key)
        keys.sort()
        new_dict = OrderedDict()
        for key in keys:
            new_dict[key] = dc[key]
        return new_dict

    true_result = sort_dict(true_result)
    predict_result = sort_dict(predict_result)

    return true_result, predict_result


def deal_xml_folders(true_folder, predict_folder, name_maps=None, iou_threshold=0.7):
    """
    compare annotations in two floders and  return two dicts
    :param true_folder: a floder contains labeled annotation
    :param predict_folder: a floder contains predicted annotation
    :param name_maps: sku map brand dictionary if no need ,use the default None
    :return: true_result(a dictionary contains label's annotation info)
            and  prediction_result(a dictionary contains prediction's annotation info)
    """
    true_xmls = os.listdir(true_folder)
    true_xmls = [file for file in true_xmls if file.endswith(".xml")]
    predict_xmls = os.listdir(predict_folder)
    predict_xmls = [file for file in predict_xmls if file.endswith(".xml")]
    true_result = {}
    predict_result = {}
    for xml in tqdm(predict_xmls, desc="deal_xml_folders"):
        if xml not in true_xmls:
            continue
        true_xml_path = os.path.join(true_folder, xml)
        predict_xml_path = os.path.join(predict_folder, xml)
        true_result, predict_result = deal_single_pair_xml(true_xml_path, predict_xml_path, iou_threshold,
                                                           name_maps, true_result, predict_result)

    return true_result, predict_result


def write_info_to_txt(result_judge_by_ground_truth, result_judge_by_predict, file_path):
    """
    write result info to file
    :param result_judge_by_ground_truth: get by deal_XML_Floders() or deal_single_pair_xml()
    :param result_judge_by_predict:  get by deal_XML_Floders() or deal_single_pair_xml()
    :param file_path:
    :return:
    """
    with open(file_path, 'w') as write:
        write.write("result info：\n\n\n")
        max_sku_name_len=0
        for k in result_judge_by_ground_truth:
              if len(k)>max_sku_name_len: max_sku_name_len=len(k)
        total_ground_truth_correct_num = 0
        total_ground_truth_false_num = 0
        total_predict_correct_num = 0
        total_predict_false_num = 0
        for k, v in result_judge_by_ground_truth.items():
            total_ground_truth_correct_num+=v['correct_num']
            total_ground_truth_false_num+=v['false_num']
            try:
                recall = v['correct_num'] / (v['correct_num'] + v['false_num']) * 100
            except ZeroDivisionError:
                recall=0
            recall = float("%.2f" % recall)
            sku_name=k.ljust(max_sku_name_len+3,' ')
            recall_info=('recall:' + str(recall) + "%").ljust(17,' ')
            ground_truth_info=("ground_truth:" + str(v)).ljust(50,' ')
            precision_info=''
            predict_info=''
            if k in result_judge_by_predict:
                predict_v = result_judge_by_predict[k]
                total_predict_correct_num+=predict_v['correct_num']
                total_predict_false_num+=predict_v['false_num']
                try:
                    precision = predict_v['correct_num'] / (predict_v['correct_num'] + predict_v['false_num']) * 100
                except ZeroDivisionError:
                    precision = 0
                precision = float("%.2f" % precision)
                precision_info=('precision:' + str(precision) + "%").ljust(20,' ')
                predict_info=("   predict:" + str(predict_v)).ljust(50,' ')
            try:
                f1 = 2 / (1 / recall + 1 / precision)
            except ZeroDivisionError:
                f1 = 0
            f1=float("%.2f" % f1)
            f1_info=('f1:' + str(f1) + "%").ljust(15,' ')
            info = sku_name+f1_info + recall_info+precision_info+ground_truth_info+predict_info
            write.write(info + '\n')
        try:
            total_recall = float("%.2f " % (total_ground_truth_correct_num / (
                        total_ground_truth_correct_num + total_ground_truth_false_num) * 100))
        except ZeroDivisionError:
            total_recall = 0
        try:
            total_precision = float(
                "%.2f " % (total_predict_correct_num / (total_predict_correct_num + total_predict_false_num) * 100))
        except ZeroDivisionError:
            total_precision = 0
        try:
            total_f1 = float("%.2f " % (2 / (1 / total_recall + 1 / total_precision)))
        except ZeroDivisionError:
            total_f1 = 0
        total_info='\n\n\ntotal_recall:{}%  total_precision:{}%    total_f1:{}%\n'.format(total_recall,total_precision,total_f1)
        write.write(total_info)


def draw_error_on_img(img_path, true_xml_file_path, predict_xml_file_path, name_maps=None, iou_threshold=0.7):
    """
    draw errors info on a new img(endwith "origin img-draw.jpg")

    yellow box means that two boxes match but name different, red font is labled type and blue font is predicted type
    red box means that labled box can't math
    blue box means that prediction box can't math
    :param img_path: the origin img(it won't be changed),and the drawed img will be in the same floder with img_path
    :param true_xml_file_path:label's annotation xml file
    :param predict_xml_file_path: prediction xml
    :param name_maps: sku map brand dictionary if no need ,use the default None
    :param iou_threshold:
    :return:
    """
    boxes_test = _phrase_xml(true_xml_file_path, name_maps, True)
    boxes_prediction = _phrase_xml(predict_xml_file_path)
    boxes_test, boxes_prediction, match_type_error, test_box_error, predict_box_error = \
        _compare_boxes(boxes_test, boxes_prediction, iou_threshold=iou_threshold)
    if len(match_type_error) == len(test_box_error) == len(predict_box_error) == 0:
        return None
    return _draw_boxes(img_path, match_type_error, test_box_error, predict_box_error)


def draw_all_bad_cases(img_dir, ground_truth_ann_dir, predict_ann_dir, bad_cases_dir,
                       name_maps=None, iou_threshold=0.7):
    """ Draw all bad cases
    :param img_dir:
    :param ground_truth_ann_dir:
    :param predict_ann_dir:
    :param bad_cases_dir:
    :param name_maps:
    :param iou_threshold:
    :return:
    """
    if not os.path.exists(bad_cases_dir):
        os.makedirs(bad_cases_dir)

    img_names = [x for x in os.listdir(img_dir) if x.lower().rsplit(".", 1)[-1] in ('jpg', 'jpeg', 'png')]
    ground_truth_ann_names = [x for x in os.listdir(ground_truth_ann_dir) if x.lower().endswith('.xml')]
    predict_ann_names = [x for x in os.listdir(predict_ann_dir) if x.lower().endswith('.xml')]
    for img_name in tqdm(img_names, desc="draw_bad_cases"):
        name_head = img_name.rsplit(".", 1)[0]
        ann_name = name_head + ".xml"
        if ann_name not in ground_truth_ann_names:
            print("Missing ground truth annotation: <%s>" % ann_name)
            continue
        if ann_name not in predict_ann_names:
            print("Missing predict annotation: <%s>" % ann_name)
            continue
        img_path = os.path.join(img_dir, img_name)
        gt_ann_path = os.path.join(ground_truth_ann_dir, ann_name)
        prd_ann_path = os.path.join(predict_ann_dir, ann_name)
        bad_cases_img = draw_error_on_img(img_path, gt_ann_path, prd_ann_path, name_maps, iou_threshold)
        if bad_cases_img:
            out_img_path = os.path.join(bad_cases_dir, img_name)
            bad_cases_img.save(out_img_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("grt_ann_path_or_dir",
                        help="The path or directory of ground truth annotations")
    parser.add_argument("prd_ann_path_or_dir",
                        help="The path or directory of predict annotations")
    parser.add_argument("result_path", nargs='?', default="./result.txt",
                        help="The path of analysis result. Default: './result.txt'")
    parser.add_argument("iou_threshold", nargs='?', default=0.7, type=float,
                        help="the box IOU threshold. Default: 0.7")
    parser.add_argument("--show_bad_cases", default=False, action='store_true',
                        help="Determine show bad cases or not. If 'show_bad_cases' is True, "
                             "args 'img_path_or_dir' and 'bad_cases_dir' should be set")
    parser.add_argument("--img_path_or_dir",
                        help="Source image(s) path or directory")
    parser.add_argument("--bad_cases_dir",
                        help="The directory of bad cases")
    _args = parser.parse_args()
    if _args.show_bad_cases:
        if _args.img_path_or_dir == '' or _args.bad_cases_dir == '':
            raise ValueError("args 'img_path_or_dir' and 'bad_cases_dir' "
                             "should be set when 'show_bad_cases' is True")
        elif _args.img_path_or_dir == _args.bad_cases_dir:
            raise ValueError("argument 'img_path_or_dir' and 'bad_cases_dir' "
                             "should be different")
    return _args


if __name__ == '__main__':
    args = get_args()

    class_name_maps = None
    if os.path.exists("classname2brand_map.json"):
        class_name_maps = load_json("classname2brand_map.json")

    true_xml_path = args.grt_ann_path_or_dir
    predict_xml_path = args.prd_ann_path_or_dir
    result_path = args.result_path
    if not (os.path.exists(true_xml_path) and os.path.exists(predict_xml_path)):
        print("please check the pathes whether really exist!")
        sys.exit()

    if os.path.isdir(true_xml_path) and os.path.isdir(predict_xml_path):
        ground_truth_result, prediction_result = deal_xml_folders(
            true_folder=args.grt_ann_path_or_dir,
            predict_folder=args.prd_ann_path_or_dir,
            name_maps=class_name_maps, iou_threshold=args.iou_threshold)
    else:
        if os.path.isfile(true_xml_path) and os.path.isfile(predict_xml_path):
            ground_truth_result, prediction_result = deal_single_pair_xml(
                true_xml_file_path=args.grt_ann_path_or_dir,
                predict_xml_file_path=args.prd_ann_path_or_dir,
                name_maps=class_name_maps, iou_threshold=args.iou_threshold)
        else:
            print("please check the XML pathes: one is folder but another is file ")
            sys.exit()

    write_info_to_txt(ground_truth_result, prediction_result, file_path=args.result_path)

    # show_bad_cases
    if args.show_bad_cases:
         draw_all_bad_cases(img_dir=args.img_path_or_dir, ground_truth_ann_dir=args.grt_ann_path_or_dir,
                            predict_ann_dir=args.prd_ann_path_or_dir, bad_cases_dir=args.bad_cases_dir,
                            name_maps=class_name_maps, iou_threshold=args.iou_threshold)
