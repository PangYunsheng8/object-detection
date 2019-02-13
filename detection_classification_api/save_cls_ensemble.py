# encoding=utf-8

"""
Multi classify model assembling
"""

import os
import json

import tensorflow as tf

build_tensor_info = tf.saved_model.utils.build_tensor_info

FLAG = tf.app.flags.FLAGS
tf.app.flags._global_parser.add_argument("--classify_model_path",
                                         default=[], nargs="+",
                                         help="Classification model paths.")
tf.app.flags.DEFINE_string("export_dir", "",
                           "Exported directory")
tf.app.flags.DEFINE_string("tensor_prefix", "",
                           "The prefix of input and output tensor")


def load_graph(graph_def_path, input_map=None):
    graph = tf.get_default_graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_def_path, "rb") as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='', input_map=input_map)
    return graph


def load_frozen_classify_graph(graph_def_path, input_map,
                               input_output_tensor_prefix=""):
    # add prefix to input tensors
    if isinstance(input_map, dict) and input_output_tensor_prefix:
        input_map_ = {input_output_tensor_prefix + k: v for k, v in input_map.items()}
        input_map = input_map_

    classify_graph = load_graph(graph_def_path, input_map=input_map)
    # get_classify_graph_inputs_outputs
    input_images = classify_graph.get_tensor_by_name(input_output_tensor_prefix + "input_images:0")
    predict = classify_graph.get_tensor_by_name(input_output_tensor_prefix + "predict:0")
    logits = classify_graph.get_tensor_by_name(input_output_tensor_prefix + "logits:0")
    outputs = {"logits": logits, "predict": predict}

    return classify_graph, input_images, outputs


def get_input_output_tensor_prefix(graph_def_path):
    prefix = ""
    info_path = os.path.join(os.path.dirname(graph_def_path), "model_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            model_info = json.load(f)
        prefix = model_info.get("input_output_tensor_prefix", "")
    return prefix


def build_assembled_classification(classify_graph_def_paths,
                                   classification_class_index_map,
                                   input_output_tensor_prefix=""):
    graph = tf.Graph()
    with graph.as_default():
        encoded_image = tf.placeholder(tf.string, name=input_output_tensor_prefix + "encoded_image")
        image_tensor = tf.image.decode_image(encoded_image, channels=3)
        shp = tf.shape(image_tensor)
        input_image_tensor = tf.reshape(image_tensor, tf.stack([-1, shp[-3], shp[-2], 3]),
                                        name=input_output_tensor_prefix + "input_images")

        # classification phase
        all_probability = []
        all_features = []
        for cls_graph_def_path in classify_graph_def_paths:
            prefix = get_input_output_tensor_prefix(cls_graph_def_path)
            classify_graph_, c_input_images_, c_outputs_ = \
                load_frozen_classify_graph(cls_graph_def_path,
                                           input_map={"input_images": input_image_tensor},
                                           input_output_tensor_prefix=prefix)
            probability = tf.nn.softmax(c_outputs_["logits"])
            all_probability.append(probability)
            all_features.append(c_outputs_["features"])
        probability = tf.reduce_mean(tf.stack(all_probability, axis=0), axis=0)
        features = tf.reduce_mean(tf.stack(all_features, axis=0), axis=0)
        c_outputs = {"scores": tf.reduce_max(probability, axis=-1),
                     "predict": tf.argmax(probability, axis=-1),
                     "features": features}

        c_logits = tf.identity(tf.log(1e-9 + probability), name=input_output_tensor_prefix + "logits")
        c_classes = tf.identity(c_outputs["predict"], name=input_output_tensor_prefix + "predict")
        c_scores = tf.identity(c_outputs["scores"], name=input_output_tensor_prefix + "scores")
        c_features = tf.identity(c_outputs["features"], name=input_output_tensor_prefix + "features")

        # add class index map
        fake_input = tf.placeholder(tf.int32)
        cls_class2index_map = {key: tf.constant(val) for key, val in classification_class_index_map.items()}

        inputs = {
            "classify_encoded_image": encoded_image,
            "classify_image_tensor": input_image_tensor,
            "fake_input": fake_input
        }
        outputs = {
            "classify_scores": c_scores,
            "classify_classes": c_classes,
            "classify_features": c_features,
            "classify_class_index_map": cls_class2index_map
        }

    return graph, inputs, outputs


def save_assembled_classify_model(graph, inputs, outputs, saved_model_path):
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_path)
    with tf.Session(graph=graph) as sess:

        # build tensor info of inputs outputs
        for key in inputs:
            inputs[key] = build_tensor_info(inputs[key])
        for key in outputs:
            if isinstance(outputs[key], tf.Tensor):
                outputs[key] = build_tensor_info(outputs[key])
            elif isinstance(outputs[key], dict):
                outputs[key] = {k: build_tensor_info(v) for k, v in outputs[key].items()}
            else:
                raise ValueError("Unexpected outputs type, only accept tf.Tensor or a dict as string to tf.Tensor map.")

        # classify only
        classify_only_inputs_tensor_info = {
            'inputs': inputs["classify_image_tensor"]
        }
        classify_only_outputs_tensor_info = {
            "scores": outputs["classify_scores"],
            "classes": outputs["classify_classes"],
            "features":outputs["classify_features"]
        }
        classify_only_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=classify_only_inputs_tensor_info,
                outputs=classify_only_outputs_tensor_info,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        # class2index_map
        fake_inputs_tensor_info = {
            'inputs': inputs["fake_input"]
        }
        classify_class2index_map_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=fake_inputs_tensor_info,
                outputs=outputs["classify_class_index_map"],
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                "classify_only": classify_only_signature,
                "classify_only_class2index_map": classify_class2index_map_signature
            },
        )
        builder.save()


def _completing_model_path(model_path_or_dir):
    return os.path.join(model_path_or_dir, "frozen_inference_graph.pb") if \
        os.path.isdir(model_path_or_dir) else model_path_or_dir


def main(_):
    assert FLAG.classify_model_path != []
    assert FLAG.export_dir != ""

    FLAG.classify_model_path = [_completing_model_path(x) for x in FLAG.classify_model_path]
    prefixes = [get_input_output_tensor_prefix(x) for x in FLAG.classify_model_path]
    t = [idx for idx, x in enumerate(prefixes) if x == FLAG.tensor_prefix]
    if len(t) > 0:
        raise Exception("The model input output tensor prefixes is same as '%s', both are '%s'. Please make them be "
                        "different." % (FLAG.classify_model_path[t[0]], FLAG.tensor_prefix))

    classify_label_index_map_path = \
        os.path.join(os.path.dirname(FLAG.classify_model_path[0]), "label_index.map")
    with open(classify_label_index_map_path) as f:
        classify_label_index_map = json.load(f)

    graph, inputs, outputs = \
        build_assembled_classification(FLAG.classify_model_path, classify_label_index_map,
                                       FLAG.tensor_prefix)

    serialized_graph = graph.as_graph_def().SerializeToString()
    save_assembled_classify_model(graph, inputs, outputs, FLAG.export_dir)
    with tf.gfile.GFile(os.path.join(FLAG.export_dir, "frozen_inference_graph.pb"), "wb") as f:
        f.write(serialized_graph)
    with open(os.path.join(FLAG.export_dir, "label_index.map"), "w") as f:
        json.dump(classify_label_index_map, f, indent=4)
    with open(os.path.join(FLAG.export_dir, "model_info.json"), "w") as f:
        model_info = {
            "input_output_tensor_prefix": FLAG.tensor_prefix
        }
        json.dump(model_info, f, indent=4, sort_keys=True)
    print("Exported model to: %s" % FLAG.export_dir)


if __name__ == '__main__':
    tf.app.run()
