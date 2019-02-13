#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-9-20 上午10:35
# @Author  : Jh Zhao
# @Site    : 
# @File    : evaluate.py
# @Software: PyCharm Community Edition

from __future__ import division

import collections
import os
import argparse
import logging
import json
import re

from lxml import etree
import numpy as np
from PIL import Image
from PIL import ImageFile
import tqdm

import detection_api
import det_cls_api

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ObjectDetectionEvaluator(object):
    """A class to evaluate detections."""

    def __init__(self,
                 categories,
                 matching_iou_threshold=0.5,
                 metric_prefix=None,
                 use_weighted_mean_ap=False):
        """Constructor.

        Args:
          categories: A list of dicts, each of which has the following keys -
            'id': (required) an integer id uniquely identifying this category.
            'name': (required) string representing category name e.g., 'cat', 'dog'.
          matching_iou_threshold: IOU threshold to use for matching groundtruth
            boxes to detection boxes.
          metric_prefix: (optional) string prefix for metric name; if None, no
            prefix is used.
          use_weighted_mean_ap: (optional) boolean which determines if the mean
            average precision is computed directly from the scores and tp_fp_labels
            of all classes.
        """
        # super(ObjectDetectionEvaluator, self).__init__(categories)
        self._categories = categories
        self._category_index = {cat['id']: cat for cat in categories}
        self._num_classes = max([cat['id'] for cat in categories])
        self._matching_iou_threshold = matching_iou_threshold
        self._use_weighted_mean_ap = use_weighted_mean_ap
        self._label_id_offset = 1
        self._evaluation = ObjectDetectionEvaluation(
            self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset)
        self._image_ids = set([])
        self._metric_prefix = (metric_prefix + '/') if metric_prefix else ''

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        Args:
          image_id: A unique string/integer identifier for the image.
          groundtruth_dict: A dictionary containing -
            "boxes": float32 numpy array
              of shape [num_boxes, 4] containing `num_boxes` groundtruth boxes of
              the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
            "classes": integer numpy array
              of shape [num_boxes] containing 1-indexed groundtruth classes for the
              boxes.

        Raises:
          ValueError: On adding groundtruth for an image more than once.
        """
        if image_id in self._image_ids:
            raise ValueError('Image with id {} already added.'.format(image_id))

        groundtruth_classes = groundtruth_dict["classes"]
        groundtruth_classes -= self._label_id_offset
        groundtruth_difficult = None
        if not len(self._image_ids) % 1000:
            logging.warn(
                'image %s does not have groundtruth difficult flag specified',
                image_id)
        self._evaluation.add_single_ground_truth_image_info(
            image_id,
            groundtruth_dict["boxes"],
            groundtruth_classes,
            groundtruth_is_difficult_list=groundtruth_difficult)
        self._image_ids.update([image_id])

    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.

        Args:
          image_id: A unique string/integer identifier for the image.
          detections_dict: A dictionary containing -
            "boxes": float32 numpy
              array of shape [num_boxes, 4] containing `num_boxes` detection boxes
              of the format [ymin, xmin, ymax, xmax] in absolute image coordinates.
            "scores": float32 numpy
              array of shape [num_boxes] containing detection scores for the boxes.
            "classes": integer numpy
              array of shape [num_boxes] containing 1-indexed detection classes for
              the boxes.
        """
        detection_classes = detections_dict[
            "classes"]
        detection_classes -= self._label_id_offset
        self._evaluation.add_single_detected_image_info(
            image_id,
            detections_dict["boxes"],
            detections_dict["scores"],
            detection_classes)

    def evaluate(self):
        """Compute evaluation result.

        Returns:
          A dictionary of metrics with the following fields -

          1. summary_metrics:
            'Precision/mAP@<matching_iou_threshold>IOU': mean average precision at
            the specified IOU threshold.

          2. per_category_ap: category specific results with keys of the form
            'PerformanceByCategory/mAP@<matching_iou_threshold>IOU/category'.
        """
        (per_class_ap, mean_ap, _, _, per_class_corloc, mean_corloc) = (
            self._evaluation.evaluate())
        pascal_metrics = {}
        for idx in range(per_class_ap.size):
            if idx + self._label_id_offset in self._category_index:
                display_name = self._category_index[idx + self._label_id_offset]['name']
                pascal_metrics[display_name] = per_class_ap[idx]

        return mean_ap, pascal_metrics

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._evaluation = ObjectDetectionEvaluation(
            self._num_classes,
            matching_iou_threshold=self._matching_iou_threshold,
            use_weighted_mean_ap=self._use_weighted_mean_ap,
            label_id_offset=self._label_id_offset)
        self._image_ids.clear()


class PascalDetectionEvaluator(ObjectDetectionEvaluator):
    """A class to evaluate detections using PASCAL metrics."""

    def __init__(self, categories, matching_iou_threshold=0.5):
        super(PascalDetectionEvaluator, self).__init__(
            categories,
            matching_iou_threshold=matching_iou_threshold,
            metric_prefix='',
            use_weighted_mean_ap=False)


def calculate_detection_result(groundtruth_dict_list, detection_dict_list):
    assert len(groundtruth_dict_list) == len(detection_dict_list), "num of two list must be same"
    classes_set = set()
    for gt in groundtruth_dict_list:
        classes_set.update(set(gt.get("classes", [])))
    for dd in detection_dict_list:
        classes_set.update(set(dd.get("classes", [])))
    categories = [{'id': i + 1, 'name': v} for i, v in enumerate(list(classes_set))]
    lbl2idx = {x['name']: x['id'] for x in categories}

    evaluator = PascalDetectionEvaluator(categories, matching_iou_threshold=0.5)

    for i, gt in enumerate(groundtruth_dict_list):
        gt['classes'] = np.array([lbl2idx[x] for x in gt.get('classes', [])], dtype=int)
        evaluator.add_single_ground_truth_image_info(i, gt)
    for i, dd in enumerate(detection_dict_list):
        dd['classes'] = np.array([lbl2idx[x] for x in dd.get('classes', [])], dtype=int)
        evaluator.add_single_detected_image_info(i, dd)

    res = evaluator.evaluate()

    return res


ObjectDetectionEvalMetrics = collections.namedtuple(
    'ObjectDetectionEvalMetrics', [
        'average_precisions', 'mean_ap', 'precisions', 'recalls', 'corlocs',
        'mean_corloc'
    ])


class ObjectDetectionEvaluation(object):
    """Internal implementation of Pascal object detection metrics."""

    def __init__(self,
                 num_groundtruth_classes,
                 matching_iou_threshold=0.5,
                 nms_iou_threshold=1.0,
                 nms_max_output_boxes=10000,
                 use_weighted_mean_ap=False,
                 label_id_offset=0):
        self.per_image_eval = PerImageEvaluation(
            num_groundtruth_classes, matching_iou_threshold, nms_iou_threshold,
            nms_max_output_boxes)
        self.num_class = num_groundtruth_classes
        self.label_id_offset = label_id_offset

        self.groundtruth_boxes = {}
        self.groundtruth_class_labels = {}
        self.groundtruth_is_difficult_list = {}
        self.groundtruth_is_group_of_list = {}
        self.num_gt_instances_per_class = np.zeros(self.num_class, dtype=int)
        self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

        self.detection_keys = set()
        self.scores_per_class = [[] for _ in range(self.num_class)]
        self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
        self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
        self.average_precision_per_class = np.empty(self.num_class, dtype=float)
        self.average_precision_per_class.fill(np.nan)
        self.precisions_per_class = []
        self.recalls_per_class = []
        self.corloc_per_class = np.ones(self.num_class, dtype=float)

        self.use_weighted_mean_ap = use_weighted_mean_ap

    def clear_detections(self):
        self.detection_keys = {}
        self.scores_per_class = [[] for _ in range(self.num_class)]
        self.tp_fp_labels_per_class = [[] for _ in range(self.num_class)]
        self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
        self.average_precision_per_class = np.zeros(self.num_class, dtype=float)
        self.precisions_per_class = []
        self.recalls_per_class = []
        self.corloc_per_class = np.ones(self.num_class, dtype=float)

    def add_single_ground_truth_image_info(self,
                                           image_key,
                                           groundtruth_boxes,
                                           groundtruth_class_labels,
                                           groundtruth_is_difficult_list=None,
                                           groundtruth_is_group_of_list=None):
        """Adds groundtruth for a single image to be used for evaluation.

        Args:
          image_key: A unique string/integer identifier for the image.
          groundtruth_boxes: float32 numpy array of shape [num_boxes, 4]
            containing `num_boxes` groundtruth boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
          groundtruth_class_labels: integer numpy array of shape [num_boxes]
            containing 0-indexed groundtruth classes for the boxes.
          groundtruth_is_difficult_list: A length M numpy boolean array denoting
            whether a ground truth box is a difficult instance or not. To support
            the case that no boxes are difficult, it is by default set as None.
          groundtruth_is_group_of_list: A length M numpy boolean array denoting
              whether a ground truth box is a group-of box or not. To support
              the case that no boxes are groups-of, it is by default set as None.
        """
        if image_key in self.groundtruth_boxes:
            logging.warn(
                'image %s has already been added to the ground truth database.',
                image_key)
            return

        self.groundtruth_boxes[image_key] = groundtruth_boxes
        self.groundtruth_class_labels[image_key] = groundtruth_class_labels
        if groundtruth_is_difficult_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_difficult_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_difficult_list[
            image_key] = groundtruth_is_difficult_list.astype(dtype=bool)
        if groundtruth_is_group_of_list is None:
            num_boxes = groundtruth_boxes.shape[0]
            groundtruth_is_group_of_list = np.zeros(num_boxes, dtype=bool)
        self.groundtruth_is_group_of_list[
            image_key] = groundtruth_is_group_of_list.astype(dtype=bool)

        self._update_ground_truth_statistics(
            groundtruth_class_labels,
            groundtruth_is_difficult_list.astype(dtype=bool),
            groundtruth_is_group_of_list.astype(dtype=bool))

    def add_single_detected_image_info(self, image_key, detected_boxes,
                                       detected_scores, detected_class_labels):
        """Adds detections for a single image to be used for evaluation.

        Args:
          image_key: A unique string/integer identifier for the image.
          detected_boxes: float32 numpy array of shape [num_boxes, 4]
            containing `num_boxes` detection boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
          detected_scores: float32 numpy array of shape [num_boxes] containing
            detection scores for the boxes.
          detected_class_labels: integer numpy array of shape [num_boxes] containing
            0-indexed detection classes for the boxes.

        Raises:
          ValueError: if the number of boxes, scores and class labels differ in
            length.
        """
        if (len(detected_boxes) != len(detected_scores) or
                    len(detected_boxes) != len(detected_class_labels)):
            raise ValueError('detected_boxes, detected_scores and '
                             'detected_class_labels should all have same lengths. Got'
                             '[%d, %d, %d]' % len(detected_boxes),
                             len(detected_scores), len(detected_class_labels))

        if image_key in self.detection_keys:
            logging.warn(
                'image %s has already been added to the detection result database',
                image_key)
            return

        self.detection_keys.add(image_key)
        if image_key in self.groundtruth_boxes:
            groundtruth_boxes = self.groundtruth_boxes[image_key]
            groundtruth_class_labels = self.groundtruth_class_labels[image_key]
            groundtruth_is_difficult_list = self.groundtruth_is_difficult_list[
                image_key]
            groundtruth_is_group_of_list = self.groundtruth_is_group_of_list[
                image_key]
        else:
            groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
            groundtruth_class_labels = np.array([], dtype=int)
            groundtruth_is_difficult_list = np.array([], dtype=bool)
            groundtruth_is_group_of_list = np.array([], dtype=bool)
        scores, tp_fp_labels, is_class_correctly_detected_in_image = (
            self.per_image_eval.compute_object_detection_metrics(
                detected_boxes, detected_scores, detected_class_labels,
                groundtruth_boxes, groundtruth_class_labels,
                groundtruth_is_difficult_list, groundtruth_is_group_of_list))

        for i in range(self.num_class):
            if scores[i].shape[0] > 0:
                self.scores_per_class[i].append(scores[i])
                self.tp_fp_labels_per_class[i].append(tp_fp_labels[i])
        self.num_images_correctly_detected_per_class += is_class_correctly_detected_in_image

    def _update_ground_truth_statistics(self, groundtruth_class_labels,
                                        groundtruth_is_difficult_list,
                                        groundtruth_is_group_of_list):
        """Update grouth truth statitistics.

        1. Difficult boxes are ignored when counting the number of ground truth
        instances as done in Pascal VOC devkit.
        2. Difficult boxes are treated as normal boxes when computing CorLoc related
        statitistics.

        Args:
          groundtruth_class_labels: An integer numpy array of length M,
              representing M class labels of object instances in ground truth
          groundtruth_is_difficult_list: A boolean numpy array of length M denoting
              whether a ground truth box is a difficult instance or not
          groundtruth_is_group_of_list: A boolean numpy array of length M denoting
              whether a ground truth box is a group-of box or not
        """
        for class_index in range(self.num_class):
            num_gt_instances = np.sum(groundtruth_class_labels[
                                          ~groundtruth_is_difficult_list
                                          & ~groundtruth_is_group_of_list] == class_index)
            self.num_gt_instances_per_class[class_index] += num_gt_instances
            if np.any(groundtruth_class_labels == class_index):
                self.num_gt_imgs_per_class[class_index] += 1

    def evaluate(self):
        """Compute evaluation result.

        Returns:
          A named tuple with the following fields -
            average_precision: float numpy array of average precision for
                each class.
            mean_ap: mean average precision of all classes, float scalar
            precisions: List of precisions, each precision is a float numpy
                array
            recalls: List of recalls, each recall is a float numpy array
            corloc: numpy float array
            mean_corloc: Mean CorLoc score for each class, float scalar
        """
        if (self.num_gt_instances_per_class == 0).any():
            logging.warn(
                'The following classes have no ground truth examples: %s',
                np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0)) +
                self.label_id_offset)

        if self.use_weighted_mean_ap:
            all_scores = np.array([], dtype=float)
            all_tp_fp_labels = np.array([], dtype=bool)

        for class_index in range(self.num_class):
            if self.num_gt_instances_per_class[class_index] == 0:
                continue
            if not self.scores_per_class[class_index]:
                scores = np.array([], dtype=float)
                tp_fp_labels = np.array([], dtype=bool)
            else:
                scores = np.concatenate(self.scores_per_class[class_index])
                tp_fp_labels = np.concatenate(self.tp_fp_labels_per_class[class_index])
            if self.use_weighted_mean_ap:
                all_scores = np.append(all_scores, scores)
                all_tp_fp_labels = np.append(all_tp_fp_labels, tp_fp_labels)
            precision, recall = compute_precision_recall(
                scores, tp_fp_labels, self.num_gt_instances_per_class[class_index])
            self.precisions_per_class.append(precision)
            self.recalls_per_class.append(recall)
            average_precision = compute_average_precision(precision, recall)
            self.average_precision_per_class[class_index] = average_precision

        self.corloc_per_class = compute_cor_loc(
            self.num_gt_imgs_per_class,
            self.num_images_correctly_detected_per_class)

        if self.use_weighted_mean_ap:
            num_gt_instances = np.sum(self.num_gt_instances_per_class)
            precision, recall = compute_precision_recall(
                all_scores, all_tp_fp_labels, num_gt_instances)
            mean_ap = compute_average_precision(precision, recall)
        else:
            mean_ap = np.nanmean(self.average_precision_per_class)
        mean_corloc = np.nanmean(self.corloc_per_class)
        return ObjectDetectionEvalMetrics(
            self.average_precision_per_class, mean_ap, self.precisions_per_class,
            self.recalls_per_class, self.corloc_per_class, mean_corloc)


# =============================== #
# ============ metrics ========== #
# =============================== #

def compute_precision_recall(scores, labels, num_gt):
    """Compute precision and recall.

    Args:
      scores: A float numpy array representing detection score
      labels: A boolean numpy array representing true/false positive labels
      num_gt: Number of ground truth instances

    Raises:
      ValueError: if the input is not of the correct format

    Returns:
      precision: Fraction of positive instances over detected ones. This value is
        None if no ground truth labels are present.
      recall: Fraction of detected positive instance over all positive instances.
        This value is None if no ground truth labels are present.

    """
    if not isinstance(
            labels, np.ndarray) or labels.dtype != np.bool or len(labels.shape) != 1:
        raise ValueError("labels must be single dimension bool numpy array")

    if not isinstance(
            scores, np.ndarray) or len(scores.shape) != 1:
        raise ValueError("scores must be single dimension numpy array")

    if num_gt < np.sum(labels):
        raise ValueError("Number of true positives must be smaller than num_gt.")

    if len(scores) != len(labels):
        raise ValueError("scores and labels must be of the same size.")

    if num_gt == 0:
        return None, None

    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    labels = labels.astype(int)
    true_positive_labels = labels[sorted_indices]
    false_positive_labels = 1 - true_positive_labels
    cum_true_positives = np.cumsum(true_positive_labels)
    cum_false_positives = np.cumsum(false_positive_labels)
    precision = cum_true_positives.astype(float) / (
        cum_true_positives + cum_false_positives)
    recall = cum_true_positives.astype(float) / num_gt
    return precision, recall


def compute_average_precision(precision, recall):
    """Compute Average Precision according to the definition in VOCdevkit.

    Precision is modified to ensure that it does not decrease as recall
    decrease.

    Args:
      precision: A float [N, 1] numpy array of precisions
      recall: A float [N, 1] numpy array of recalls

    Raises:
      ValueError: if the input is not of the correct format

    Returns:
      average_precison: The area under the precision recall curve. NaN if
        precision and recall are None.

    """
    if precision is None:
        if recall is not None:
            raise ValueError("If precision is None, recall must also be None")
        return np.NAN

    if not isinstance(precision, np.ndarray) or not isinstance(recall,
                                                               np.ndarray):
        raise ValueError("precision and recall must be numpy array")
    if precision.dtype != np.float or recall.dtype != np.float:
        raise ValueError("input must be float numpy array.")
    if len(precision) != len(recall):
        raise ValueError("precision and recall must be of the same size.")
    if not precision.size:
        return 0.0
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("Precision must be in the range of [0, 1].")
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall must be in the range of [0, 1].")
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError("recall must be a non-decreasing array")

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision


def compute_cor_loc(num_gt_imgs_per_class,
                    num_images_correctly_detected_per_class):
    """Compute CorLoc according to the definition in the following paper.

    https://www.robots.ox.ac.uk/~vgg/rg/papers/deselaers-eccv10.pdf

    Returns nans if there are no ground truth images for a class.

    Args:
      num_gt_imgs_per_class: 1D array, representing number of images containing
          at least one object instance of a particular class
      num_images_correctly_detected_per_class: 1D array, representing number of
          images that are correctly detected at least one object instance of a
          particular class

    Returns:
      corloc_per_class: A float numpy array represents the corloc score of each
        class
    """
    return np.where(
        num_gt_imgs_per_class == 0,
        np.nan,
        num_images_correctly_detected_per_class / num_gt_imgs_per_class)


# =============================== #
# ===== per_image_evaluation  === #
# =============================== #

class PerImageEvaluation(object):
    """Evaluate detection result of a single image."""

    def __init__(self,
                 num_groundtruth_classes,
                 matching_iou_threshold=0.5,
                 nms_iou_threshold=0.3,
                 nms_max_output_boxes=50):
        """Initialized PerImageEvaluation by evaluation parameters.

        Args:
          num_groundtruth_classes: Number of ground truth object classes
          matching_iou_threshold: A ratio of area intersection to union, which is
              the threshold to consider whether a detection is true positive or not
          nms_iou_threshold: IOU threshold used in Non Maximum Suppression.
          nms_max_output_boxes: Number of maximum output boxes in NMS.
        """
        self.matching_iou_threshold = matching_iou_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_max_output_boxes = nms_max_output_boxes
        self.num_groundtruth_classes = num_groundtruth_classes

    def compute_object_detection_metrics(
            self, detected_boxes, detected_scores, detected_class_labels,
            groundtruth_boxes, groundtruth_class_labels,
            groundtruth_is_difficult_lists, groundtruth_is_group_of_list):
        """Evaluates detections as being tp, fp or ignored from a single image.

        The evaluation is done in two stages:
         1. All detections are matched to non group-of boxes; true positives are
            determined and detections matched to difficult boxes are ignored.
         2. Detections that are determined as false positives are matched against
            group-of boxes and ignored if matched.

        Args:
          detected_boxes: A float numpy array of shape [N, 4], representing N
              regions of detected object regions.
              Each row is of the format [y_min, x_min, y_max, x_max]
          detected_scores: A float numpy array of shape [N, 1], representing
              the confidence scores of the detected N object instances.
          detected_class_labels: A integer numpy array of shape [N, 1], repreneting
              the class labels of the detected N object instances.
          groundtruth_boxes: A float numpy array of shape [M, 4], representing M
              regions of object instances in ground truth
          groundtruth_class_labels: An integer numpy array of shape [M, 1],
              representing M class labels of object instances in ground truth
          groundtruth_is_difficult_lists: A boolean numpy array of length M denoting
              whether a ground truth box is a difficult instance or not
          groundtruth_is_group_of_list: A boolean numpy array of length M denoting
              whether a ground truth box has group-of tag

        Returns:
          scores: A list of C float numpy arrays. Each numpy array is of
              shape [K, 1], representing K scores detected with object class
              label c
          tp_fp_labels: A list of C boolean numpy arrays. Each numpy array
              is of shape [K, 1], representing K True/False positive label of
              object instances detected with class label c
          is_class_correctly_detected_in_image: a numpy integer array of
              shape [C, 1], indicating whether the correponding class has a least
              one instance being correctly detected in the image
        """
        detected_boxes, detected_scores, detected_class_labels = (
            self._remove_invalid_boxes(detected_boxes, detected_scores,
                                       detected_class_labels))
        scores, tp_fp_labels = self._compute_tp_fp(
            detected_boxes, detected_scores, detected_class_labels,
            groundtruth_boxes, groundtruth_class_labels,
            groundtruth_is_difficult_lists, groundtruth_is_group_of_list)

        is_class_correctly_detected_in_image = self._compute_cor_loc(
            detected_boxes, detected_scores, detected_class_labels,
            groundtruth_boxes, groundtruth_class_labels)
        return scores, tp_fp_labels, is_class_correctly_detected_in_image

    def _compute_cor_loc(self, detected_boxes, detected_scores,
                         detected_class_labels, groundtruth_boxes,
                         groundtruth_class_labels):
        """Compute CorLoc score for object detection result.

        Args:
          detected_boxes: A float numpy array of shape [N, 4], representing N
              regions of detected object regions.
              Each row is of the format [y_min, x_min, y_max, x_max]
          detected_scores: A float numpy array of shape [N, 1], representing
              the confidence scores of the detected N object instances.
          detected_class_labels: A integer numpy array of shape [N, 1], repreneting
              the class labels of the detected N object instances.
          groundtruth_boxes: A float numpy array of shape [M, 4], representing M
              regions of object instances in ground truth
          groundtruth_class_labels: An integer numpy array of shape [M, 1],
              representing M class labels of object instances in ground truth
        Returns:
          is_class_correctly_detected_in_image: a numpy integer array of
              shape [C, 1], indicating whether the correponding class has a least
              one instance being correctly detected in the image
        """
        is_class_correctly_detected_in_image = np.zeros(
            self.num_groundtruth_classes, dtype=int)
        for i in range(self.num_groundtruth_classes):
            gt_boxes_at_ith_class = groundtruth_boxes[groundtruth_class_labels ==
                                                      i, :]
            detected_boxes_at_ith_class = detected_boxes[detected_class_labels ==
                                                         i, :]
            detected_scores_at_ith_class = detected_scores[detected_class_labels == i]
            is_class_correctly_detected_in_image[i] = (
                self._compute_is_aclass_correctly_detected_in_image(
                    detected_boxes_at_ith_class, detected_scores_at_ith_class,
                    gt_boxes_at_ith_class))

        return is_class_correctly_detected_in_image

    def _compute_is_aclass_correctly_detected_in_image(
            self, detected_boxes, detected_scores, groundtruth_boxes):
        """Compute CorLoc score for a single class.

        Args:
          detected_boxes: A numpy array of shape [N, 4] representing detected box
              coordinates
          detected_scores: A 1-d numpy array of length N representing classification
              score
          groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
              box coordinates

        Returns:
          is_class_correctly_detected_in_image: An integer 1 or 0 denoting whether a
              class is correctly detected in the image or not
        """
        if detected_boxes.size > 0:
            if groundtruth_boxes.size > 0:
                max_score_id = np.argmax(detected_scores)
                detected_boxlist = BoxList(
                    np.expand_dims(detected_boxes[max_score_id, :], axis=0))
                gt_boxlist = BoxList(groundtruth_boxes)
                iou = np_box_list_ops_iou(detected_boxlist, gt_boxlist)
                if np.max(iou) >= self.matching_iou_threshold:
                    return 1
        return 0

    def _compute_tp_fp(self, detected_boxes, detected_scores,
                       detected_class_labels, groundtruth_boxes,
                       groundtruth_class_labels, groundtruth_is_difficult_lists,
                       groundtruth_is_group_of_list):
        """Labels true/false positives of detections of an image across all classes.

        Args:
          detected_boxes: A float numpy array of shape [N, 4], representing N
              regions of detected object regions.
              Each row is of the format [y_min, x_min, y_max, x_max]
          detected_scores: A float numpy array of shape [N, 1], representing
              the confidence scores of the detected N object instances.
          detected_class_labels: A integer numpy array of shape [N, 1], repreneting
              the class labels of the detected N object instances.
          groundtruth_boxes: A float numpy array of shape [M, 4], representing M
              regions of object instances in ground truth
          groundtruth_class_labels: An integer numpy array of shape [M, 1],
              representing M class labels of object instances in ground truth
          groundtruth_is_difficult_lists: A boolean numpy array of length M denoting
              whether a ground truth box is a difficult instance or not
          groundtruth_is_group_of_list: A boolean numpy array of length M denoting
              whether a ground truth box has group-of tag

        Returns:
          result_scores: A list of float numpy arrays. Each numpy array is of
              shape [K, 1], representing K scores detected with object class
              label c
          result_tp_fp_labels: A list of boolean numpy array. Each numpy array is of
              shape [K, 1], representing K True/False positive label of object
              instances detected with class label c
        """
        result_scores = []
        result_tp_fp_labels = []
        for i in range(self.num_groundtruth_classes):
            gt_boxes_at_ith_class = groundtruth_boxes[(groundtruth_class_labels == i
                                                       ), :]
            groundtruth_is_difficult_list_at_ith_class = (
                groundtruth_is_difficult_lists[groundtruth_class_labels == i])
            groundtruth_is_group_of_list_at_ith_class = (
                groundtruth_is_group_of_list[groundtruth_class_labels == i])
            detected_boxes_at_ith_class = detected_boxes[(detected_class_labels == i
                                                          ), :]
            detected_scores_at_ith_class = detected_scores[detected_class_labels == i]
            scores, tp_fp_labels = self._compute_tp_fp_for_single_class(
                detected_boxes_at_ith_class, detected_scores_at_ith_class,
                gt_boxes_at_ith_class, groundtruth_is_difficult_list_at_ith_class,
                groundtruth_is_group_of_list_at_ith_class)
            result_scores.append(scores)
            result_tp_fp_labels.append(tp_fp_labels)
        return result_scores, result_tp_fp_labels

    def _remove_invalid_boxes(self, detected_boxes, detected_scores,
                              detected_class_labels):
        valid_indices = np.logical_and(detected_boxes[:, 0] < detected_boxes[:, 2],
                                       detected_boxes[:, 1] < detected_boxes[:, 3])
        return (detected_boxes[valid_indices, :], detected_scores[valid_indices],
                detected_class_labels[valid_indices])

    def _compute_tp_fp_for_single_class(
            self, detected_boxes, detected_scores, groundtruth_boxes,
            groundtruth_is_difficult_list, groundtruth_is_group_of_list):
        """Labels boxes detected with the same class from the same image as tp/fp.

        Args:
          detected_boxes: A numpy array of shape [N, 4] representing detected box
              coordinates
          detected_scores: A 1-d numpy array of length N representing classification
              score
          groundtruth_boxes: A numpy array of shape [M, 4] representing ground truth
              box coordinates
          groundtruth_is_difficult_list: A boolean numpy array of length M denoting
              whether a ground truth box is a difficult instance or not. If a
              groundtruth box is difficult, every detection matching this box
              is ignored.
          groundtruth_is_group_of_list: A boolean numpy array of length M denoting
              whether a ground truth box has group-of tag. If a groundtruth box
              is group-of box, every detection matching this box is ignored.

        Returns:
          Two arrays of the same size, containing all boxes that were evaluated as
          being true positives or false positives; if a box matched to a difficult
          box or to a group-of box, it is ignored.

          scores: A numpy array representing the detection scores.
          tp_fp_labels: a boolean numpy array indicating whether a detection is a
              true positive.

        """
        if detected_boxes.size == 0:
            return np.array([], dtype=float), np.array([], dtype=bool)
        detected_boxlist = BoxList(detected_boxes)
        detected_boxlist.add_field('scores', detected_scores)
        detected_boxlist = np_box_list_ops_non_max_suppression(
            detected_boxlist, self.nms_max_output_boxes, self.nms_iou_threshold)

        scores = detected_boxlist.get_field('scores')

        if groundtruth_boxes.size == 0:
            return scores, np.zeros(detected_boxlist.num_boxes(), dtype=bool)

        tp_fp_labels = np.zeros(detected_boxlist.num_boxes(), dtype=bool)
        is_matched_to_difficult_box = np.zeros(
            detected_boxlist.num_boxes(), dtype=bool)
        is_matched_to_group_of_box = np.zeros(
            detected_boxlist.num_boxes(), dtype=bool)

        # The evaluation is done in two stages:
        # 1. All detections are matched to non group-of boxes; true positives are
        #    determined and detections matched to difficult boxes are ignored.
        # 2. Detections that are determined as false positives are matched against
        #    group-of boxes and ignored if matched.

        # Tp-fp evaluation for non-group of boxes (if any).
        gt_non_group_of_boxlist = BoxList(
            groundtruth_boxes[~groundtruth_is_group_of_list, :])
        if gt_non_group_of_boxlist.num_boxes() > 0:
            groundtruth_nongroup_of_is_difficult_list = groundtruth_is_difficult_list[
                ~groundtruth_is_group_of_list]
            iou = np_box_list_ops_iou(detected_boxlist, gt_non_group_of_boxlist)
            max_overlap_gt_ids = np.argmax(iou, axis=1)
            is_gt_box_detected = np.zeros(
                gt_non_group_of_boxlist.num_boxes(), dtype=bool)
            for i in range(detected_boxlist.num_boxes()):
                gt_id = max_overlap_gt_ids[i]
                if iou[i, gt_id] >= self.matching_iou_threshold:
                    if not groundtruth_nongroup_of_is_difficult_list[gt_id]:
                        if not is_gt_box_detected[gt_id]:
                            tp_fp_labels[i] = True
                            is_gt_box_detected[gt_id] = True
                    else:
                        is_matched_to_difficult_box[i] = True

        # Tp-fp evaluation for group of boxes.
        gt_group_of_boxlist = BoxList(
            groundtruth_boxes[groundtruth_is_group_of_list, :])
        if gt_group_of_boxlist.num_boxes() > 0:
            ioa = np_box_list_ops_ioa(gt_group_of_boxlist, detected_boxlist)
            max_overlap_group_of_gt = np.max(ioa, axis=0)
            for i in range(detected_boxlist.num_boxes()):
                if (not tp_fp_labels[i] and not is_matched_to_difficult_box[i] and
                            max_overlap_group_of_gt[i] >= self.matching_iou_threshold):
                    is_matched_to_group_of_box[i] = True

        return scores[~is_matched_to_difficult_box
                      & ~is_matched_to_group_of_box], tp_fp_labels[
                   ~is_matched_to_difficult_box
                   & ~is_matched_to_group_of_box]


# =============================== #
# ========== np BoxList  ======== #
# =============================== #

class BoxList(object):
    """Box collection.

    BoxList represents a list of bounding boxes as numpy array, where each
    bounding box is represented as a row of 4 numbers,
    [y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes within a
    given list correspond to a single image.

    Optionally, users can add additional related fields (such as
    objectness/classification scores).
    """

    def __init__(self, data):
        """Constructs box collection.

        Args:
          data: a numpy array of shape [N, 4] representing box coordinates

        Raises:
          ValueError: if bbox data is not a numpy array
          ValueError: if invalid dimensions for bbox data
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array.')
        if len(data.shape) != 2 or data.shape[1] != 4:
            raise ValueError('Invalid dimensions for box data.')
        if data.dtype != np.float32 and data.dtype != np.float64:
            raise ValueError('Invalid data type for box data: float is required.')
        if not self._is_valid_boxes(data):
            raise ValueError('Invalid box data. data must be a numpy array of '
                             'N*[y_min, x_min, y_max, x_max]')
        self.data = {'boxes': data}

    def num_boxes(self):
        """Return number of boxes held in collections."""
        return self.data['boxes'].shape[0]

    def get_extra_fields(self):
        """Return all non-box fields."""
        return [k for k in self.data.keys() if k != 'boxes']

    def has_field(self, field):
        return field in self.data

    def add_field(self, field, field_data):
        """Add data to a specified field.

        Args:
          field: a string parameter used to speficy a related field to be accessed.
          field_data: a numpy array of [N, ...] representing the data associated
              with the field.
        Raises:
          ValueError: if the field is already exist or the dimension of the field
              data does not matches the number of boxes.
        """
        if self.has_field(field):
            raise ValueError('Field ' + field + 'already exists')
        if len(field_data.shape) < 1 or field_data.shape[0] != self.num_boxes():
            raise ValueError('Invalid dimensions for field data')
        self.data[field] = field_data

    def get(self):
        """Convenience function for accesssing box coordinates.

        Returns:
          a numpy array of shape [N, 4] representing box corners
        """
        return self.get_field('boxes')

    def get_field(self, field):
        """Accesses data associated with the specified field in the box collection.

        Args:
          field: a string parameter used to speficy a related field to be accessed.

        Returns:
          a numpy 1-d array representing data of an associated field

        Raises:
          ValueError: if invalid field
        """
        if not self.has_field(field):
            raise ValueError('field {} does not exist'.format(field))
        return self.data[field]

    def get_coordinates(self):
        """Get corner coordinates of boxes.

        Returns:
         a list of 4 1-d numpy arrays [y_min, x_min, y_max, x_max]
        """
        box_coordinates = self.get()
        y_min = box_coordinates[:, 0]
        x_min = box_coordinates[:, 1]
        y_max = box_coordinates[:, 2]
        x_max = box_coordinates[:, 3]
        return [y_min, x_min, y_max, x_max]

    def _is_valid_boxes(self, data):
        """Check whether data fullfills the format of N*[ymin, xmin, ymax, xmin].

        Args:
          data: a numpy array of shape [N, 4] representing box coordinates

        Returns:
          a boolean indicating whether all ymax of boxes are equal or greater than
              ymin, and all xmax of boxes are equal or greater than xmin.
        """
        if data.shape[0] > 0:
            for i in range(data.shape[0]):
                if data[i, 0] > data[i, 2] or data[i, 1] > data[i, 3]:
                    return False
        return True


# =============================== #
# ======= np_box_list_ops  ====== #
# =============================== #

class SortOrder(object):
    """Enum class for sort order.

    Attributes:
      ascend: ascend order.
      descend: descend order.
    """
    ASCEND = 1
    DESCEND = 2


def np_box_list_ops_area(boxlist):
    """Computes area of boxes.

    Args:
      boxlist: BoxList holding N boxes

    Returns:
      a numpy array with shape [N*1] representing box areas
    """
    y_min, x_min, y_max, x_max = boxlist.get_coordinates()
    return (y_max - y_min) * (x_max - x_min)


def np_box_list_ops_intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes

    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area
    """
    return np_box_ops_intersection(boxlist1.get(), boxlist2.get())


def np_box_list_ops_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes

    Returns:
      a numpy array with shape [N, M] representing pairwise iou scores.
    """
    return np_box_ops_iou(boxlist1.get(), boxlist2.get())


def np_box_list_ops_ioa(boxlist1, boxlist2):
    """Computes pairwise intersection-over-area between box collections.

    Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, IOA(box1, box2) != IOA(box2, box1).

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes

    Returns:
      a numpy array with shape [N, M] representing pairwise ioa scores.
    """
    return np_box_ops_ioa(boxlist1.get(), boxlist2.get())


def np_box_list_ops_gather(boxlist, indices, fields=None):
    """Gather boxes from BoxList according to indices and return new BoxList.

    By default, Gather returns boxes corresponding to the input index list, as
    well as all additional fields stored in the boxlist (indexing into the
    first dimension).  However one can optionally only gather from a
    subset of fields.

    Args:
      boxlist: BoxList holding N boxes
      indices: a 1-d numpy array of type int_
      fields: (optional) list of fields to also gather from.  If None (default),
          all fields are gathered from.  Pass an empty fields list to only gather
          the box coordinates.

    Returns:
      subboxlist: a BoxList corresponding to the subset of the input BoxList
          specified by indices

    Raises:
      ValueError: if specified field is not contained in boxlist or if the
          indices are not of type int_
    """
    if indices.size:
        if np.amax(indices) >= boxlist.num_boxes() or np.amin(indices) < 0:
            raise ValueError('indices are out of valid range.')
    subboxlist = BoxList(boxlist.get()[indices, :])
    if fields is None:
        fields = boxlist.get_extra_fields()
    for field in fields:
        extra_field_data = boxlist.get_field(field)
        subboxlist.add_field(field, extra_field_data[indices, ...])
    return subboxlist


def np_box_list_ops_sort_by_field(boxlist, field, order=SortOrder.DESCEND):
    """Sort boxes and associated fields according to a scalar field.

    A common use case is reordering the boxes according to descending scores.

    Args:
      boxlist: BoxList holding N boxes.
      field: A BoxList field for sorting and reordering the BoxList.
      order: (Optional) 'descend' or 'ascend'. Default is descend.

    Returns:
      sorted_boxlist: A sorted BoxList with the field in the specified order.

    Raises:
      ValueError: if specified field does not exist or is not of single dimension.
      ValueError: if the order is not either descend or ascend.
    """
    if not boxlist.has_field(field):
        raise ValueError('Field ' + field + ' does not exist')
    if len(boxlist.get_field(field).shape) != 1:
        raise ValueError('Field ' + field + 'should be single dimension.')
    if order != SortOrder.DESCEND and order != SortOrder.ASCEND:
        raise ValueError('Invalid sort order')

    field_to_sort = boxlist.get_field(field)
    sorted_indices = np.argsort(field_to_sort)
    if order == SortOrder.DESCEND:
        sorted_indices = sorted_indices[::-1]
    return np_box_list_ops_gather(boxlist, sorted_indices)


def np_box_list_ops_non_max_suppression(boxlist,
                                        max_output_size=10000,
                                        iou_threshold=1.0,
                                        score_threshold=-10.0):
    """Non maximum suppression.

    This op greedily selects a subset of detection bounding boxes, pruning
    away boxes that have high IOU (intersection over union) overlap (> thresh)
    with already selected boxes. In each iteration, the detected bounding box with
    highest score in the available pool is selected.

    Args:
      boxlist: BoxList holding N boxes.  Must contain a 'scores' field
        representing detection scores. All scores belong to the same class.
      max_output_size: maximum number of retained boxes
      iou_threshold: intersection over union threshold.
      score_threshold: minimum score threshold. Remove the boxes with scores
                       less than this value. Default value is set to -10. A very
                       low threshold to pass pretty much all the boxes, unless
                       the user sets a different score threshold.

    Returns:
      a BoxList holding M boxes where M <= max_output_size
    Raises:
      ValueError: if 'scores' field does not exist
      ValueError: if threshold is not in [0, 1]
      ValueError: if max_output_size < 0
    """
    if not boxlist.has_field('scores'):
        raise ValueError('Field scores does not exist')
    if iou_threshold < 0. or iou_threshold > 1.0:
        raise ValueError('IOU threshold must be in [0, 1]')
    if max_output_size < 0:
        raise ValueError('max_output_size must be bigger than 0.')

    boxlist = np_box_list_ops_filter_scores_greater_than(boxlist, score_threshold)
    if boxlist.num_boxes() == 0:
        return boxlist

    boxlist = np_box_list_ops_sort_by_field(boxlist, 'scores')

    # Prevent further computation if NMS is disabled.
    if iou_threshold == 1.0:
        if boxlist.num_boxes() > max_output_size:
            selected_indices = np.arange(max_output_size)
            return np_box_list_ops_gather(boxlist, selected_indices)
        else:
            return boxlist

    boxes = boxlist.get()
    num_boxes = boxlist.num_boxes()
    # is_index_valid is True only for all remaining valid boxes,
    is_index_valid = np.full(num_boxes, 1, dtype=bool)
    selected_indices = []
    num_output = 0
    for i in range(num_boxes):
        if num_output < max_output_size:
            if is_index_valid[i]:
                num_output += 1
                selected_indices.append(i)
                is_index_valid[i] = False
                valid_indices = np.where(is_index_valid)[0]
                if valid_indices.size == 0:
                    break

                intersect_over_union = np_box_ops_iou(
                    np.expand_dims(boxes[i, :], axis=0), boxes[valid_indices, :])
                intersect_over_union = np.squeeze(intersect_over_union, axis=0)
                is_index_valid[valid_indices] = np.logical_and(
                    is_index_valid[valid_indices],
                    intersect_over_union <= iou_threshold)
    return np_box_list_ops_gather(boxlist, np.array(selected_indices))


def np_box_list_ops_filter_scores_greater_than(boxlist, thresh):
    """Filter to keep only boxes with score exceeding a given threshold.

    This op keeps the collection of boxes whose corresponding scores are
    greater than the input threshold.

    Args:
      boxlist: BoxList holding N boxes.  Must contain a 'scores' field
        representing detection scores.
      thresh: scalar threshold

    Returns:
      a BoxList holding M boxes where M <= N

    Raises:
      ValueError: if boxlist not a BoxList object or if it does not
        have a scores field
    """
    if not isinstance(boxlist, BoxList):
        raise ValueError('boxlist must be a BoxList')
    if not boxlist.has_field('scores'):
        raise ValueError('input boxlist must have \'scores\' field')
    scores = boxlist.get_field('scores')
    if len(scores.shape) > 2:
        raise ValueError('Scores should have rank 1 or 2')
    if len(scores.shape) == 2 and scores.shape[1] != 1:
        raise ValueError('Scores should have rank 1 or have shape '
                         'consistent with [None, 1]')
    high_score_indices = np.reshape(np.where(np.greater(scores, thresh)),
                                    [-1]).astype(np.int32)
    return np_box_list_ops_gather(boxlist, high_score_indices)


# =============================== #
# ========== np_box_ops  ======== #
# =============================== #

def np_box_ops_area(boxes):
    """Computes area of boxes.

    Args:
      boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
      a numpy array with shape [N*1] representing box areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def np_box_ops_intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes
      boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def np_box_ops_iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes.
      boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
      a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = np_box_ops_intersection(boxes1, boxes2)
    area1 = np_box_ops_area(boxes1)
    area2 = np_box_ops_area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(
        area2, axis=0) - intersect
    return intersect / union


def np_box_ops_ioa(boxes1, boxes2):
    """Computes pairwise intersection-over-area between box collections.

    Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, IOA(box1, box2) != IOA(box2, box1).

    Args:
      boxes1: a numpy array with shape [N, 4] holding N boxes.
      boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
      a numpy array with shape [N, M] representing pairwise ioa scores.
    """
    intersect = np_box_ops_intersection(boxes1, boxes2)
    areas = np.expand_dims(np_box_ops_area(boxes2), axis=0)
    return intersect / areas


# ================================================== #
# ================================================== #
# ================================================== #
def _recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` and `point` tags are the only twos that can appear
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
        if child.tag not in ('object', 'point'):
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def _read_annotation(annotation_path):
    with open(annotation_path, 'rb') as f:
        xml_str = f.read()
    xml = etree.fromstring(xml_str)
    data = _recursive_parse_xml_to_dict(xml)["annotation"]
    return data


_RESOURCES = {}


def detection_inference_one(image):
    _detector = _RESOURCES["detector"]
    if _RESOURCES["detector_type"] == "server":
        res = _detector.detection_classify(image, timeout=60)
        res["boxes"] = res["boxes"][0]
        width, height = image.size
        res["boxes"][:, [0, 2]] = height * res["boxes"][:, [0, 2]]
        res["boxes"][:, [1, 3]] = width * res["boxes"][:, [1, 3]]
    else:
        # TODO
        res = {}
    return res


def detection_inference(image_paths):
    infer_res = []
    for p in tqdm.tqdm(image_paths, desc="detection_inference"):
        img = Image.open(p)
        d = detection_inference_one(img)
        infer_res.append(d)
    return infer_res


def get_groundtruth(annotation_paths):
    groundtruth_list = []
    for p in annotation_paths:
        ann = _read_annotation(p)
        boxes = []
        classes = []
        for obj in ann.get("object", []):
            box = [float(obj['bndbox'][x]) for x in ['ymin', 'xmin', 'ymax', 'xmax']]
            boxes.append(box)
            classes.append(obj['name'])
        gt = {"boxes": np.asarray(boxes, dtype=np.float32),
              "classes": classes}
        groundtruth_list.append(gt)
    return groundtruth_list


def get_name_head(path):
    return os.path.basename(path).split(".", 1)[0]


def evaluate(image_dir, annotation_dir):
    # collect image
    names = [x for x in os.listdir(image_dir) if x.rsplit(".", 1)[-1].lower() in ('jpg', 'jpeg', 'png')]
    heads = [get_name_head(x) for x in names]
    paths = [os.path.join(image_dir, x) for x in names]
    image_paths = dict(zip(heads, paths))
    # collect annotation
    names = [x for x in os.listdir(annotation_dir) if x.rsplit(".", 1)[-1].lower() == 'xml']
    heads = [get_name_head(x) for x in names]
    paths = [os.path.join(annotation_dir, x) for x in names]
    annotation_paths = dict(zip(heads, paths))
    # check annotation complement
    img_head_set = set(image_paths.keys())
    ann_head_set = set(annotation_paths.keys())
    no_ann = img_head_set - ann_head_set
    if no_ann:
        for x in list(no_ann)[:10]:
            logging.warning("with no annotation of %s" % x)
    no_img = ann_head_set - img_head_set
    if no_img:
        for x in list(no_img)[:10]:
            logging.warning("with no annotation of %s" % x)
    inter = img_head_set.intersection(ann_head_set)
    inter = list(inter)
    image_paths = [image_paths[x] for x in inter]
    annotation_paths = [annotation_paths[x] for x in inter]

    # inference image
    prediction_list = detection_inference(image_paths)
    # read annotations
    groundtruth_list = get_groundtruth(annotation_paths)

    eval_result = calculate_detection_result(groundtruth_list, prediction_list)

    return eval_result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True,
                        help="image directory")
    parser.add_argument("--annotation_dir",
                        help="annotation directory, it will be same as image_dir when unset")
    parser.add_argument("--model_path", required=True,
                        help="model path, like 'server:ip:port/model_name' or 'file:/home/*/to/model/path'")

    _args = parser.parse_args()
    if not _args.annotation_dir:
        _args.annotation_dir = _args.image_dir
    if not (re.match("^server:[0-9a-zA-Z_.]+:\d+/[0-9a-zA-Z_.]+$", _args.model_path)
            or re.match("^file:.+$", _args.model_path)):
        raise ValueError("model path should be like 'server:ip:port/model_name' or 'file:/home/*/to/model/path'")

    return _args


if __name__ == '__main__':
    args = get_args()
    if args.model_path.startswith("server:"):
        host, port, model_name = re.findall("server:([0-9a-zA-Z_.]+):(\d+)/([0-9a-zA-Z_.]+)$", args.model_path)[0]
        _RESOURCES["detector"] = detection_api.ColaDetector(host, port, model_name)
        _RESOURCES["detector_type"] = "server"
    else:
        model_path = re.findall("^file:(.+)$", args.model_path)[0]
        _RESOURCES["detector"] = det_cls_api.ColaDetectorClassify(model_path)
        _RESOURCES["detector_type"] = "file"
    map_, aps = evaluate(args.image_dir, args.annotation_dir)
    print("mAP: %s" % map_)
    print(json.dumps(aps, indent=4, ensure_ascii=False))
