# encoding=utf-8

import os
import re
import json
from collections import defaultdict

import numpy as np
import argparse

np.random.seed(0)

_IMAGE_DATASETS = {}


def get_image_dataset(dataset_root, shuffle=False):
    if dataset_root not in _IMAGE_DATASETS:
        if os.path.isfile(dataset_root):
            with open(dataset_root) as f:
                _IMAGE_DATASETS[dataset_root] = json.load(f)
        else:
            class_names = os.listdir(dataset_root)
            class_folders = [os.path.join(dataset_root, x) for x in class_names]
            class_names_folders = [x for x in zip(class_names, class_folders) if os.path.isdir(x[1])]

            image_paths, image_classes = [], []
            for cls_name, cls_folder in class_names_folders:
                image_names = os.listdir(cls_folder)
                image_names = [x for x in image_names if x[-4:] in (".jpg", ".jpeg", ".png")]
                image_paths += [os.path.join(cls_folder, x) for x in image_names]
                image_classes += [cls_name] * len(image_names)

            if shuffle:
                shf_idx = np.random.permutation(len(image_paths))
                image_paths = [image_paths[i] for i in shf_idx]
                image_classes = [image_classes[i] for i in shf_idx]

            _IMAGE_DATASETS[dataset_root] = {'paths': image_paths, 'classes': image_classes}

    return _IMAGE_DATASETS[dataset_root]


def dataset_slice(dataset, indices):
    ds = {}
    for k in dataset:
        ds[k] = [dataset[k][i] for i in indices]
    return ds


def dataset_split(dataset, part_ratios, output_path):
    assert sum(part_ratios) == 1 and all([r > 0 for r in part_ratios])

    cls_indices = defaultdict(list)
    for i, cls in enumerate(dataset['classes']):
        cls_indices[cls].append(i)

    result_indices = [[] for _ in range(len(part_ratios))]

    for cls, idc in cls_indices.items():
        pre_end = 0
        next_start = 0
        for idx, ratio in enumerate(part_ratios):
            part_num = int(round(len(idc) * ratio))
            next_start += part_num
            result_indices[idx] += idc[pre_end:next_start]
            pre_end = next_start

    result_datasets = [dataset_slice(dataset, idc) for idc in result_indices]

    pre_end = 0
    next_start = 0
    for ds, ratio in zip(result_datasets, part_ratios):
        next_start += ratio
        path = output_path + "_%03d-%03d_of_100" % \
                             (int(round(100 * pre_end)), int(round(100 * next_start)))
        pre_end = next_start
        with open(path, 'w') as f:
            json.dump(ds, f, indent=2)


def dataset_split_kfold(dataset, k, output_path):
    image_paths, image_classes = dataset['paths'], dataset['classes']
    num_cases = len(image_paths)
    assert 0 < k <= num_cases
    part_ratios = np.arange(0, 1, 1 / k)
    part_ratios = np.concatenate((part_ratios, [1]))

    cls_indices = defaultdict(list)
    for i, cls in enumerate(dataset['classes']):
        cls_indices[cls].append(i)

    for idx, ratio in enumerate(part_ratios[:-1]):
        train_idc = []
        test_idc = []
        for img_cls, idc in cls_indices.items():
            num_cases_cls = len(idc)
            start = int(round(num_cases_cls * ratio))
            end = int(round(num_cases_cls * part_ratios[idx + 1]))
            train_idc += idc[:start] + idc[end:]
            test_idc += idc[start:end]
        train_ds = dataset_slice(dataset, train_idc)
        test_ds = dataset_slice(dataset, test_idc)

        train_data_path = output_path + "_%02d-of-%02d-fold_train" % (idx, k)
        with open(train_data_path, 'w') as f:
            json.dump(train_ds, f, indent=2)
        test_data_path = output_path + "_%02d-of-%02d-fold_test" % (idx, k)
        with open(test_data_path, 'w') as f:
            json.dump(test_ds, f, indent=2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dataset", help="image dataset directory")
    parser.add_argument("--ratios", default='1.', help="split ratios, like '0.8,0.1,0.1' ")
    parser.add_argument("--output_path", help="the output files path")
    parser.add_argument("--k_fold", default=False, action='store_true',
                        help='output k fold part dataset, when this is true you must set the argument k.')
    parser.add_argument("--k", type=int, help='the number of k-fold')
    args_ = parser.parse_args()

    if args_.k_fold:
        if not isinstance(args_.k, int) or args_.k <= 0:
            raise ValueError("argument k is None or non-positive, please set a valid k.")
    else:
        if not re.match("^\d*(\.\d*)?(,\d*(\.\d*)?)*,?$", args_.ratios):
            raise SyntaxError("ratios should be like '0.8,0.1,0.1' ")
        args_.ratios = eval(args_.ratios)
        if not isinstance(args_.ratios, (tuple, list)):
            args_.ratios = (args_.ratios,)
        if max(args_.ratios) > 1 and min(args_.ratios) <= 0:
            raise ValueError('ratios should be bigger than 0 and no bigger than 1')
        if sum(args_.ratios) + 1e-12 < 1:
            args_.ratios = args_.ratios + (1 - sum(args_.ratios),)
        else:
            ratios = args_.ratios[:-1] + (1 - sum(args_.ratios[:-1]),)
            args_.ratios = tuple(ratios)

    return args_


if __name__ == '__main__':
    args = get_args()
    image_dataset = get_image_dataset(args.image_dataset, shuffle=True)
    if args.k_fold:
        dataset_split_kfold(image_dataset, args.k, args.output_path)
    else:
        dataset_split(image_dataset, args.ratios, args.output_path)
