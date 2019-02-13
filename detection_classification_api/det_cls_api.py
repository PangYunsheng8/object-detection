# encoding=utf-8

import os
import json

import numpy as np
import tensorflow as tf

__all__ = [
    "ColaDetector", "ColaClassifier", "ColaDetectorClassify"
]


def load_graph(graph_def_path, input_map=None):
    graph = tf.get_default_graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_def_path, "rb") as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='', input_map=input_map)
    return graph


def load_frozen_detection_graph(graph_def_path, input_map=None,
                                input_output_tensor_prefix=""):
    # add prefix to input tensors
    if isinstance(input_map, dict) and input_output_tensor_prefix:
        input_map_ = {input_output_tensor_prefix + k: v for k, v in input_map.items()}
        input_map = input_map_

    detection_graph = load_graph(graph_def_path, input_map=input_map)
    # get_detection_graph_inputs_outputs
    inputs_names = [
        "image_tensor"
    ]
    outputs_names = [
        "detection_boxes",
        "detection_scores",
        "detection_classes",
        "num_detections"
    ]
    inputs = {name: detection_graph.get_tensor_by_name(input_output_tensor_prefix + name + ":0")
              for name in inputs_names}
    outputs = {name: detection_graph.get_tensor_by_name(input_output_tensor_prefix + name + ":0")
               for name in outputs_names}
    return detection_graph, inputs, outputs


def load_frozen_classify_graph(graph_def_path, input_map=None,
                               input_output_tensor_prefix=""):
    # add prefix to input tensors
    if isinstance(input_map, dict) and input_output_tensor_prefix:
        input_map_ = {input_output_tensor_prefix + k: v for k, v in input_map.items()}
        input_map = input_map_

    classify_graph = load_graph(graph_def_path, input_map=input_map)
    # get_classify_graph_inputs_outputs
    inputs_names = [
        "input_images"
    ]
    outputs_names = [
        "predict",
        "logits"
    ]
    inputs = {name: classify_graph.get_tensor_by_name(input_output_tensor_prefix + name + ":0")
              for name in inputs_names}
    outputs = {name: classify_graph.get_tensor_by_name(input_output_tensor_prefix + name + ":0")
               for name in outputs_names}
    return classify_graph, inputs, outputs


def load_frozen_detection_classify_graph(graph_def_path, input_map=None,
                                         input_output_tensor_prefix=""):
    # add prefix to input tensors
    if isinstance(input_map, dict) and input_output_tensor_prefix:
        input_map_ = {input_output_tensor_prefix + k: v for k, v in input_map.items()}
        input_map = input_map_

    dc_graph = load_graph(graph_def_path, input_map=input_map)
    # get_detection_graph_inputs_outputs
    inputs_names = [
        "detection_encoded_image",
        "detection_image_tensor",
        "classify_image_tensor"
    ]
    outputs_names = [
        "classify_scores",
        "classify_classes",
        "detection_only_boxes",
        "detection_only_scores",
        "detection_only_classes",
        "detection_classify_scores",
        "detection_classify_classes"
    ]
    inputs = {name: dc_graph.get_tensor_by_name(input_output_tensor_prefix + name + ":0")
              for name in inputs_names}
    outputs = {name: dc_graph.get_tensor_by_name(input_output_tensor_prefix + name + ":0")
               for name in outputs_names}
    return dc_graph, inputs, outputs


def load_label_index_map(label_index_map_path):
    with open(label_index_map_path) as f:
        label_index_map = json.load(f)
    if isinstance(label_index_map, list):
        label_index_map = {x['class']: x for x in label_index_map}
    else:
        label_index_map = {k: {"id": i, 'class': k} for k, i in label_index_map.items()}
    index_label_map = {v['id']: v for k, v in label_index_map.items()}
    return label_index_map, index_label_map


def get_input_output_tensor_prefix(graph_def_path):
    prefix = ""
    info_path = os.path.join(os.path.dirname(graph_def_path), "model_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            model_info = json.load(f)
        prefix = model_info.get("input_output_tensor_prefix", "")
    return prefix


class ColaDetector(object):
    def __init__(self, frozen_model_path, label_index_map_path=None):
        if os.path.isdir(frozen_model_path):
            model_dir = frozen_model_path
            frozen_model_path = os.path.join(frozen_model_path, "frozen_inference_graph.pb")
        else:
            model_dir = os.path.dirname(frozen_model_path)
        if label_index_map_path is None or label_index_map_path == "":
            label_index_map_path = os.path.join(model_dir, "label_index.map")

        prefix = get_input_output_tensor_prefix(frozen_model_path)
        classify_graph, inputs, outputs = load_frozen_detection_graph(frozen_model_path,
                                                                      input_output_tensor_prefix=prefix)
        label_index_map, index_label_map = load_label_index_map(label_index_map_path)
        self.classify_graph = classify_graph
        self.inputs = inputs
        self.outputs = outputs
        self.model_signature = {
            "input_image_tensor": self.inputs.get("image_tensor"),
            "outputs_tensor": self.outputs
        }
        self.label_index_map = label_index_map
        self.index_label_map = index_label_map
        self.session = tf.Session(graph=classify_graph)

    def detect(self, images):
        images = np.asarray(images).astype(np.uint8)
        assert 4 >= images.ndim >= 3 == images.shape[-1]
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        outputs = self.session.run(self.model_signature["outputs_tensor"],
                                   feed_dict={self.model_signature["input_image_tensor"]: images})
        return outputs

    def get_detection_result(self, image, threshold=0.7, return_label_name=False):
        predict = self.detect(image)
        boxes = predict["detection_boxes"]
        scores = predict["detection_scores"]
        classes = predict["detection_classes"]
        # num = predict["num_detections"]

        idx = np.argmax(scores < threshold)
        boxes = boxes[:, :idx, ...]
        scores = scores[:, :idx, ...]
        classes = classes[:, :idx, ...]
        num = idx
        if return_label_name:
            classes = [[self.index_label_map[x] for x in y] for y in classes]
            classes = [[x['display_name'] if 'display_name' in x else x['class'] for x in y] for y in classes]
        return boxes, scores, classes, num

    def get_detection_patches(self, image, threshold=0.7):
        boxes, scores, classes, num = self.get_detection_result(image, threshold, True)
        image_arr = np.asarray(image)
        height, width = image_arr.shape[-3:-1]
        ird = lambda x: int(round(x))
        boxes = [[ird(x[0] / height), ird(x[1] / width), ird(x[2] / height), ird(x[3] / width)] for x in boxes]
        patches = []
        for bx, sc, cl in zip(boxes, scores, classes):
            ymin, xmin, ymax, xmax = bx
            patches.append(image[..., ymin:ymax, xmin:xmax, :])
        return patches, classes, scores


class ColaClassifier(object):
    def __init__(self, frozen_model_path, label_index_map_path=None):
        if os.path.isdir(frozen_model_path):
            model_dir = frozen_model_path
            frozen_model_path = os.path.join(frozen_model_path, "frozen_inference_graph.pb")
        else:
            model_dir = os.path.dirname(frozen_model_path)
        if label_index_map_path is None or label_index_map_path == "":
            label_index_map_path = os.path.join(model_dir, "label_index.map")

        prefix = get_input_output_tensor_prefix(frozen_model_path)
        classify_graph, input_images, outputs = load_frozen_classify_graph(frozen_model_path,
                                                                           input_output_tensor_prefix=prefix)
        label_index_map, index_label_map = load_label_index_map(label_index_map_path)
        self.classify_graph = classify_graph
        self.inputs = input_images
        self.outputs = outputs
        self.model_signature = {
            "input_image_tensor": self.inputs.get("input_images"),
            "outputs_tensor": self.outputs
        }
        self.label_index_map = label_index_map
        self.index_label_map = index_label_map
        self.session = tf.Session(graph=classify_graph)

    def predict_images(self, images):
        images = np.asarray(images).astype(np.uint8)
        assert 4 >= images.ndim >= 3 == images.shape[-1]
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        outputs = self.session.run(self.model_signature["outputs_tensor"],
                                   feed_dict={self.model_signature["input_image_tensor"]: images})
        return outputs

    def get_top_1(self, image, return_label_name=False):
        predict = self.predict_images(image)
        prd = predict["predict"]
        if return_label_name:
            prd = [self.index_label_map[x] for x in prd]
            prd = [x['display_name'] if 'display_name' in x else x['class'] for x in prd]
        return prd[0]

    def get_top_n(self, image, n=5, return_label_name=False):
        assert 0 < n <= len(self.index_label_map)
        predict = self.predict_images(image)
        lgt = predict["logits"]
        top_n = np.argsort(lgt, axis=-1)[..., -n:][..., ::-1]
        if return_label_name:
            top_n = [[self.index_label_map[x] for x in y] for y in top_n]
            top_n = [[x['display_name'] if 'display_name' in x else x['class'] for x in y] for y in top_n]
        return top_n[0]


class ColaDetectorClassify(object):
    def __init__(self, frozen_model_path, label_index_map_path=None):
        if os.path.isdir(frozen_model_path):
            model_dir = frozen_model_path
            frozen_model_path = os.path.join(frozen_model_path, "frozen_inference_graph.pb")
        else:
            model_dir = os.path.dirname(frozen_model_path)
        if label_index_map_path is None or label_index_map_path == "":
            label_index_map_path = os.path.join(model_dir, "label_index.map")

        prefix = get_input_output_tensor_prefix(frozen_model_path)
        dc_graph, inputs, outputs = load_frozen_detection_classify_graph(frozen_model_path,
                                                                         input_output_tensor_prefix=prefix)
        label_index_map, index_label_map = load_label_index_map(label_index_map_path)
        self.graph = dc_graph
        self.inputs = inputs
        self.outputs = outputs
        self.detection_only_signature = {
            "input_image_tensor": self.inputs.get("detection_image_tensor"),
            "outputs_tensor": {
                "boxes": self.outputs.get("detection_only_boxes"),
                "scores": self.outputs.get("detection_only_scores"),
                "classes": self.outputs.get("detection_only_classes")
            }
        }
        self.classify_only_signature = {
            "input_image_tensor": self.inputs.get("classify_image_tensor"),
            "outputs_tensor": {
                "scores": self.outputs.get("classify_scores"),
                "classes": self.outputs.get("classify_classes")
            }
        }
        self.detection_classify_signature = {
            "input_image_tensor": self.inputs.get("detection_image_tensor"),
            "outputs_tensor": {
                "boxes": self.outputs.get("detection_only_boxes"),
                "scores": self.outputs.get("detection_classify_scores"),
                "classes": self.outputs.get("detection_classify_classes")
            }
        }
        self.label_index_map = label_index_map
        self.index_label_map = index_label_map
        self.session = tf.Session(graph=dc_graph)

    def _preprocess(self, image):
        images = np.asarray(image).astype(np.uint8)
        assert 4 >= images.ndim >= 3 == images.shape[-1]
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        return images

    def detection_only(self, image):
        images = self._preprocess(image)
        outputs = self.session.run(self.detection_only_signature["outputs_tensor"],
                                   feed_dict={self.detection_only_signature["input_image_tensor"]: images})
        return outputs

    def classify_only(self, image):
        images = self._preprocess(image)
        outputs = self.session.run(self.classify_only_signature["outputs_tensor"],
                                   feed_dict={self.classify_only_signature["input_image_tensor"]: images})
        return outputs

    def detection_classify(self, image):
        images = self._preprocess(image)
        outputs = self.session.run(self.detection_classify_signature["outputs_tensor"],
                                   feed_dict={self.detection_classify_signature["input_image_tensor"]: images})
        return outputs
