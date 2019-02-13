"""Builder for preprocessing steps."""

import tensorflow as tf

from classification.core import preprocessor
from classification.protos import preprocessor_pb2


def _get_step_config_from_proto(preprocessor_step_config, step_name):
  """Returns the value of a field named step_name from proto.

  Args:
    preprocessor_step_config: A preprocessor_pb2.PreprocessingStep object.
    step_name: Name of the field to get value from.

  Returns:
    result_dict: a sub proto message from preprocessor_step_config which will be
                 later converted to a dictionary.

  Raises:
    ValueError: If field does not exist in proto.
  """
  for field, value in preprocessor_step_config.ListFields():
    if field.name == step_name:
      return value

  raise ValueError('Could not get field %s from proto!', step_name)


def _get_dict_from_proto(config):
  """Helper function to put all proto fields into a dictionary.

  For many preprocessing steps, there's an trivial 1-1 mapping from proto fields
  to function arguments. This function automatically populates a dictionary with
  the arguments from the proto.

  Protos that CANNOT be trivially populated include:
  * nested messages.
  * steps that check if an optional field is set (ie. where None != 0).
  * protos that don't map 1-1 to arguments (ie. list should be reshaped).
  * fields requiring additional validation (ie. repeated field has n elements).

  Args:
    config: A protobuf object that does not violate the conditions above.

  Returns:
    result_dict: |config| converted into a python dictionary.
  """
  result_dict = {}
  for field, value in config.ListFields():
    result_dict[field.name] = value
  return result_dict


# A map from a PreprocessingStep proto config field name to the preprocessing
# function that should be used. The PreprocessingStep proto should be parsable
# with _get_dict_from_proto.
PREPROCESSING_FUNCTION_MAP = {
    'normalize_image': preprocessor.normalize_image,
    'random_pixel_value_scale': preprocessor.random_pixel_value_scale,
    'random_rgb_to_gray': preprocessor.random_rgb_to_gray,
    'random_adjust_brightness': preprocessor.random_adjust_brightness,
    'random_adjust_contrast': preprocessor.random_adjust_contrast,
    'random_adjust_hue': preprocessor.random_adjust_hue,
    'random_adjust_saturation': preprocessor.random_adjust_saturation,
    'random_distort_color': preprocessor.random_distort_color,
    'random_black_patches': preprocessor.random_black_patches
}


# A map to convert from preprocessor_pb2.ResizeImage.Method enum to
# tf.image.ResizeMethod.
RESIZE_METHOD_MAP = {
    preprocessor_pb2.ResizeImage.AREA: tf.image.ResizeMethod.AREA,
    preprocessor_pb2.ResizeImage.BICUBIC: tf.image.ResizeMethod.BICUBIC,
    preprocessor_pb2.ResizeImage.BILINEAR: tf.image.ResizeMethod.BILINEAR,
    preprocessor_pb2.ResizeImage.NEAREST_NEIGHBOR: (
        tf.image.ResizeMethod.NEAREST_NEIGHBOR),
}


def build(preprocessor_step_config):
  """Builds preprocessing step based on the configuration.

  Args:
    preprocessor_step_config: PreprocessingStep configuration proto.

  Returns:
    function, argmap: A callable function and an argument map to call function
                      with.

  Raises:
    ValueError: On invalid configuration.
  """
  step_type = preprocessor_step_config.WhichOneof('preprocessing_step')

  if step_type in PREPROCESSING_FUNCTION_MAP:
    preprocessing_function = PREPROCESSING_FUNCTION_MAP[step_type]
    step_config = _get_step_config_from_proto(preprocessor_step_config,
                                              step_type)
    function_args = _get_dict_from_proto(step_config)
    return (preprocessing_function, function_args)

  if step_type == 'random_horizontal_flip':
    config = preprocessor_step_config.random_horizontal_flip
    return (preprocessor.random_horizontal_flip, {})

  if step_type == 'random_vertical_flip':
    config = preprocessor_step_config.random_vertical_flip
    return (preprocessor.random_vertical_flip,{})

  if step_type == 'random_rotation90':
    return (preprocessor.random_rotation90, {})

  if step_type == 'random_crop_image':
    config = preprocessor_step_config.random_crop_image
    return (preprocessor.random_crop_image,
            {"crop_width_range": (config.width_range.start, config.width_range.end),
             "crop_height_range": (config.height_range.start, config.height_range.end),
             "crop_probability": config.crop_probability})

  if step_type == 'resize_image':
    config = preprocessor_step_config.resize_image
    method = RESIZE_METHOD_MAP[config.method]
    return (preprocessor.resize_image,
            {
                'new_height': config.new_height,
                'new_width': config.new_width,
                'method': method
            })

  raise ValueError('Unknown preprocessing step.')
