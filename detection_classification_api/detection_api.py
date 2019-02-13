# encoding=utf-8

from __future__ import print_function

from PIL import Image
from grpc.beta import implementations
import numpy
import tensorflow as tf

from protos import predict_pb2
from protos import prediction_service_pb2


def decode_result(result):
    """
    :param result:
    :return:
    """
    d_result = {}
    for key in result.outputs:
        if result.outputs[key].dtype == 1:
            vals = numpy.asarray(result.outputs[key].float_val, dtype=numpy.float32)
        elif result.outputs[key].dtype == 2:
            vals = numpy.asarray(result.outputs[key].double_val, dtype=numpy.double)
        elif result.outputs[key].dtype == 3:
            vals = result.outputs[key].int_val
        elif result.outputs[key].dtype == 4:
            vals = result.outputs[key].uint8_val
        elif result.outputs[key].dtype == 5:
            vals = result.outputs[key].int16_val
        elif result.outputs[key].dtype == 6:
            vals = result.outputs[key].int8_val
        elif result.outputs[key].dtype == 7:
            vals = result.outputs[key].string_val
            vals = [x.decode() for x in vals]
        elif result.outputs[key].dtype == 8:
            vals = result.outputs[key].complex64_val
        elif result.outputs[key].dtype == 9:
            vals = numpy.asarray(result.outputs[key].int64_val, dtype=numpy.int64)
        elif result.outputs[key].dtype == 10:
            vals = result.outputs[key].bool_val
        else:
            raise Exception("Unknown dtype.")
        shp = [dim.size for dim in result.outputs[key].tensor_shape.dim]
        d_result[key] = numpy.reshape(vals, shp)
    return d_result


class ColaDetector(object):
    def __init__(self, host, port, model_name, max_detection_size=None, max_classification_size=None):
        """
        :param host: str, serving model's host ip or dns
        :param port: integer, serving model's port
        :param model_name: str, model name
        :param max_detection_size: integer, the max size for detection
        :param max_classification_size: integer, the max size for classification
        """
        self.channel = implementations.insecure_channel(host, int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.model_name = model_name
        self.max_detection_size = max_detection_size
        self.max_classification_size = max_classification_size

    def detection_only(self, image, callback=None, timeout=30):
        """ detection only,
        :param image: Image.Image object
        :param callback: callback function
        :param timeout:
        :return:
        if callback is None, return a result dict {'scores': Arr[1*N], 'classes': Arr[N], 'boxes': Arr[1*N*4]},
        if define callback function, this function will process the upper result.
        """
        assert isinstance(image, Image.Image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.max_detection_size is not None:
            ratio = max(image.size) / self.max_detection_size
            if ratio > 1:
                image = image.resize([int(round(x / ratio)) for x in image.size])
        exp_image_arr = numpy.expand_dims(numpy.asarray(image), axis=0)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = 'detection_only_with_names'
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(exp_image_arr, shape=exp_image_arr.shape))

        def _callback(res):
            d_res = decode_result(res.result())
            return callback(d_res)

        if callback is not None:
            result_future = self.stub.Predict.future(request, timeout)
            result_future.add_done_callback(_callback)
        else:
            result_future = self.stub.Predict(request, timeout)
            result_future = decode_result(result_future)
            return result_future

    def classify_only(self, image, callback=None, timeout=30):
        """ classify only
        :param image: Image.Image object
        :param callback: callback function
        :param timeout:
        :return:
        if callback is None, return a result dict {'scores': Arr[1*1], 'classes': Arr[1]},
        if define callback function, this function will process the upper result.
        """
        assert isinstance(image, Image.Image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.max_classification_size is not None:
            ratio = max(image.size) / self.max_classification_size
            if ratio > 1:
                image = image.resize([int(round(x / ratio)) for x in image.size])
        exp_image_arr = numpy.expand_dims(numpy.asarray(image), axis=0)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = 'classify_only_with_names'
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(exp_image_arr, shape=exp_image_arr.shape))

        def _callback(res):
            d_res = decode_result(res.result())
            return callback(d_res)

        if callback is not None:
            result_future = self.stub.Predict.future(request, timeout)
            result_future.add_done_callback(_callback)
        else:
            result_future = self.stub.Predict(request, timeout)
            result_future = decode_result(result_future)
            return result_future

    def detection_classify(self, image, callback=None, timeout=30):
        """ detection only,
        :param image: Image.Image object
        :param callback: callback function
        :param timeout:
        :return:
        if callback is None, return a result dict {'scores': Arr[N], 'classes': Arr[N], 'boxes': Arr[1*N*4]},
        if define callback function, this function will process the upper result.
        """
        assert isinstance(image, Image.Image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.max_detection_size is not None:
            ratio = max(image.size) / self.max_detection_size
            if ratio > 1:
                image = image.resize([int(round(x / ratio)) for x in image.size])
        exp_image_arr = numpy.expand_dims(numpy.asarray(image), axis=0)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = 'detection_classify_with_names'
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(exp_image_arr, shape=exp_image_arr.shape))

        def _callback(res):
            d_res = decode_result(res.result())
            return callback(d_res)

        if callback is not None:
            result_future = self.stub.Predict.future(request, timeout)
            result_future.add_done_callback(_callback)
        else:
            result_future = self.stub.Predict(request, timeout)
            result_future = decode_result(result_future)
            return result_future
