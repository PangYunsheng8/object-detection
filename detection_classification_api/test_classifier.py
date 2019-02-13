# encoding=utf-8


import os
import json
import shutil

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from det_cls_api import ColaClassifier

plt.switch_backend('agg')

FLAG = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model_path", "",
                           "Path of frozen model")
tf.app.flags.DEFINE_string("test_image_dataset", "",
                           "Test image dataset folder or json file "
                           "contain test image paths and classes")
tf.app.flags.DEFINE_string("logdir", "",
                           "Train image folder")

IMAGE_DATASETS = {}


def get_image_dataset(dataset_root, shuffle=False):
    if dataset_root in IMAGE_DATASETS:
        return IMAGE_DATASETS.get(dataset_root)
    else:
        class_names = os.listdir(dataset_root)
        class_folders = [os.path.join(dataset_root, x) for x in class_names]
        class_names_folders = [x for x in zip(class_names, class_folders) if os.path.isdir(x[1])]

        image_paths, image_classes = [], []
        for cls_name, cls_folder in class_names_folders:
            image_names = os.listdir(cls_folder)
            image_names = [x for x in image_names if x[-4:] in (".jpg", ".png")]
            image_paths += [os.path.join(cls_folder, x) for x in image_names]
            image_classes += [cls_name] * len(image_names)

        if shuffle:
            image_paths, image_classes = dataset_shuffle(image_paths, image_classes)

        IMAGE_DATASETS[dataset_root] = (image_paths, image_classes)
        return image_paths, image_classes


def dataset_shuffle(image_paths, image_classes):
    shf_idx = np.random.permutation(len(image_paths))
    image_paths = [image_paths[i] for i in shf_idx]
    image_classes = [image_classes[i] for i in shf_idx]
    return image_paths, image_classes


def load_image_dataset_map(path):
    with open(path) as f:
        dataset = json.load(f)
    IMAGE_DATASETS[path] = (dataset['paths'], dataset['classes'])


def _softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    r = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return r


def test_classifier(classifier, image_paths, image_classes):
    num_classes = len(classifier.label_index_map)
    index_label_map = classifier.index_label_map

    top_1_cnt, top_k_cnt = 0, 0
    match_table = np.zeros([num_classes, num_classes], dtype=np.float32)
    bad_cases = {}
    for img_path, img_class in zip(image_paths, image_classes):
        output = classifier.predict_images(Image.open(img_path))
        prd, prob = output["predict"], _softmax(output["logits"])
        prd = np.argmax(prob, axis=1)
        t1a = 1 if img_class == prd[0] else 0
        top_5 = np.argsort(prob, axis=-1)[..., -5:][..., ::-1]
        tka = 1 if img_class in top_5 else 0
        top_1_cnt += t1a
        top_k_cnt += tka
        match_table[prd[0], img_class] += 1
        if prd[0] != img_class:
            if index_label_map[img_class]["class"] not in bad_cases:
                bad_cases[index_label_map[img_class]["class"]] = []
            sorted_prob_arg = np.argsort(prob[0])
            top_3 = list(reversed(sorted_prob_arg))[:3]
            top_3_prob = [prob[0, idx] for idx in top_3]
            top_3_cls = [index_label_map[idx]["class"] for idx in top_3]
            top_3_bad_case = list(zip(top_3_cls, top_3_prob))
            bad_cases[index_label_map[img_class]["class"]].append((img_path, list(zip(top_3, top_3_prob))))
            if len(bad_cases[index_label_map[img_class]["class"]]) <= 3:
                info = "; ".join(["%s[%.4f]" % (i, p) for i, p in top_3_bad_case])
                print("BadCases:: %s => %s" % (index_label_map[img_class]["class"], info))
    prd_correct = np.eye(num_classes, num_classes) * match_table
    precise = np.sum(prd_correct, axis=1) / (1e-8 + np.sum(match_table, axis=1))
    recall = np.sum(prd_correct, axis=0) / (1e-8 + np.sum(match_table, axis=0))
    top_1_acc = top_1_cnt / len(image_paths)
    top_k_acc = top_k_cnt / len(image_paths)

    # show test result
    print("\n")
    print("Top1_Acc: %s/%s\t%s" % (int(top_1_cnt), len(image_paths), top_1_acc))
    print("Top5_Acc: %s/%s\t%s" % (int(top_k_cnt), len(image_paths), top_k_acc))
    label_names = [index_label_map[i]["class"] for i in range(num_classes)]
    print("\n")
    line_format = "{:<32}{:<16}{:<16}{:<16}"
    print(line_format.format("class name", "precise", "recall", "f1"))
    for name, pcs, rcl in zip(label_names, precise, recall):
        f1 = 2 * pcs * rcl / (1e-8 + pcs + rcl)
        print(line_format.format(name, "%.6f" % pcs, "%.6f" % rcl, "%.6f" % f1))
    import pickle
    with open("match_table.pkl", "wb") as f:
        pickle.dump(match_table, f)
    match_table = np.log(1e-8 + match_table)
    plt.clf()
    plt.figure(figsize=(24, 24))
    ax = plt.gcf().add_subplot(1, 1, 1)
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names)
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90)
    im = ax.imshow(match_table, cmap=plt.get_cmap('hot'), interpolation='nearest',
                   vmin=0, vmax=np.max(match_table))
    ax.grid()
    plt.colorbar(im, shrink=0.2)
    fld = os.path.join(FLAG.logdir, "test")
    if not tf.gfile.IsDirectory(fld):
        tf.gfile.MakeDirs(fld)
    plt.savefig(os.path.join(fld, "match_table.jpg"))

    test_epoch_fld = os.path.join(FLAG.logdir, "test", "bad_cases")
    if not tf.gfile.IsDirectory(test_epoch_fld):
        tf.gfile.MakeDirs(test_epoch_fld)
    for class_name, vals in bad_cases.items():
        cls_fld = os.path.join(test_epoch_fld, class_name)
        if not tf.gfile.IsDirectory(cls_fld):
            tf.gfile.MakeDirs(cls_fld)
        for img_path, prd in vals:
            name = "_".join(["%02d[%.4f]" % (i, p) for i, p in prd]) + ".jpg"
            path = os.path.join(cls_fld, name)
            shutil.copyfile(img_path, path)


def main(_):
    assert FLAG.model_path != ""
    assert FLAG.test_image_dataset != ""
    assert FLAG.logdir != ""

    classifier = ColaClassifier(FLAG.model_path)

    if not tf.gfile.IsDirectory(FLAG.test_image_dataset):
        load_image_dataset_map(FLAG.test_image_dataset)
    image_paths, image_classes = get_image_dataset(FLAG.test_image_dataset)

    image_classes = [classifier.label_index_map[x]["id"] for x in image_classes]
    test_classifier(classifier, image_paths, image_classes)


if __name__ == '__main__':
    tf.app.run()
