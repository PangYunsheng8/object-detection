"""Preprocess images for classification.

We perform two sets of operations in preprocessing stage:
(a) operations that are applied to both training and testing data,
(b) operations that are applied only to training data for the purpose of
    data augmentation.

A preprocessing function receives an image,
performs an operation on them, and returns them.
Some examples are: randomly cropping the image, randomly mirroring the image,
                   randomly changing the brightness, contrast, hue.

The preprocess function receives an image tensor.
The image is a rank 3 tensor: [height, width, channels] with
dtype=tf.float32. 

"""

import sys

import tensorflow as tf

############################ normalization ##############################

def normalize_image(image,
                    original_minval=0,
                    original_maxval=255,
                    target_minval=0,
                    target_maxval=1):
  """Normalizes pixel values in the image.

  Moves the pixel values from the current [original_minval, original_maxval]
  range to a the [target_minval, target_maxval] range.

  Args:
    image: rank 3 float32 tensor containing 1
           image -> [height, width, channels].
    original_minval: current image minimum value.
    original_maxval: current image maximum value.
    target_minval: target image minimum value.
    target_maxval: target image maximum value.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('NormalizeImage', values=[image]):
    original_minval = float(original_minval)
    original_maxval = float(original_maxval)
    target_minval = float(target_minval)
    target_maxval = float(target_maxval)
    image = tf.to_float(image)
    image = tf.subtract(image, original_minval)
    image = tf.multiply(image, (target_maxval - target_minval) /
                        (original_maxval - original_minval))
    image = tf.add(image, target_minval)
    return image

################################ resize ####################################

def resize_image(image,
                 new_height=224,
                 new_width=224,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
  """See `tf.image.resize_images` for detailed doc."""
  with tf.name_scope(
      'ResizeImage',
      values=[image, new_height, new_width, method, align_corners]):
    new_image = tf.image.resize_images(
        image, [new_height, new_width],
        method=method,
        align_corners=align_corners)

    return new_image

########################### rotation / reflection ##############################

def random_horizontal_flip(image,
                           seed=None):
  """Randomly flips the image horizontally.

  The probability of flipping the image is 50%.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    seed: random seed

  Returns:
    image: image which is the same shape as input image.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  with tf.name_scope('RandomHorizontalFlip', values=[image]):
    # random variable defining whether to do flip or not
    do_a_flip_random = tf.greater(tf.random_uniform([], seed=seed), 0.5)

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)

    return image

def random_vertical_flip(image,
                         seed=None):
  """Randomly flips the image vertically.

  The probability of flipping the image is 50%.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    seed: random seed

  Returns:
    image: image which is the same shape as input image.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_up_down(image)
    return image_flipped

  with tf.name_scope('RandomVerticalFlip', values=[image]):
    # random variable defining whether to do flip or not
    do_a_flip_random = tf.greater(tf.random_uniform([], seed=seed), 0.5)

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)

    return image

def random_rotation90(image,
                      seed=None):
  """Randomly rotates the image 90 degrees counter-clockwise.

  The probability of rotating the image is 50%. This can be combined with
  random_horizontal_flip and random_vertical_flip to produce an output with a
  uniform distribution of the eight possible 90 degree rotation / reflection
  combinations.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    seed: random seed

  Returns:
    image: image which is the same shape as input image.
  """

  def _rot90_image(image):
    # rotate image
    image_rotated = tf.image.rot90(image)
    return image_rotated

  with tf.name_scope('RandomRotation90', values=[image]):

    # random variable defining whether to rotate by 90 degrees or not
    do_a_rot90_random = tf.greater(tf.random_uniform([], seed=seed), 0.5)

    # rotate image
    image = tf.cond(do_a_rot90_random, lambda: _rot90_image(image),
                    lambda: image)

    return image

############################ pixel value scale ################################

def random_pixel_value_scale(image, minval=0.9, maxval=1.1, seed=None):
  """Scales each value in the pixels of the image.

     This function scales each pixel independent of the other ones.
     For each value in image tensor, draws a random number between
     minval and maxval and multiples the values with them.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    minval: lower ratio of scaling pixel values.
    maxval: upper ratio of scaling pixel values.
    seed: random seed.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomPixelValueScale', values=[image]):
    color_coef = tf.random_uniform(
        tf.shape(image),
        minval=minval,
        maxval=maxval,
        dtype=tf.float32,
        seed=seed)
    image = tf.multiply(image, color_coef)
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image

############################ rgb to grayscale ################################

def random_rgb_to_gray(image, probability=0.1, seed=None):
  """Changes the image from RGB to Grayscale with the given probability.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    probability: the probability of returning a grayscale image.
            The probability should be a number between [0, 1].
    seed: random seed.

  Returns:
    image: image which is the same shape as input image.
  """
  def _image_to_gray(image):
    image_gray1 = tf.image.rgb_to_grayscale(image)
    image_gray3 = tf.image.grayscale_to_rgb(image_gray1)
    return image_gray3

  with tf.name_scope('RandomRGBtoGray', values=[image]):
    # random variable defining whether to do flip or not
    do_gray_random = tf.random_uniform([], seed=seed)

    image = tf.cond(
        tf.greater(do_gray_random, probability), lambda: image,
        lambda: _image_to_gray(image))

  return image

############################ color distortion ################################

def random_adjust_brightness(image, max_delta=0.2):
  """Randomly adjusts brightness.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: how much to change the brightness. A value between [0, 1).

  Returns:
    image: image which is the same shape as input image.
    boxes: boxes which is the same shape as input boxes.
  """
  with tf.name_scope('RandomAdjustBrightness', values=[image]):
    image = tf.image.random_brightness(image, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

def random_adjust_contrast(image, min_delta=0.8, max_delta=1.25):
  """Randomly adjusts contrast.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustContrast', values=[image]):
    image = tf.image.random_contrast(image, min_delta, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

def random_adjust_hue(image, max_delta=0.02):
  """Randomly adjusts hue.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: change hue randomly with a value between 0 and max_delta.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustHue', values=[image]):
    image = tf.image.random_hue(image, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

def random_adjust_saturation(image, min_delta=0.8, max_delta=1.25):
  """Randomly adjusts saturation.

  Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustSaturation', values=[image]):
    image = tf.image.random_saturation(image, min_delta, max_delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

def random_distort_color(image, color_ordering=0):
  """Randomly distorts color.

  Randomly distorts color using a combination of brightness, hue, contrast
  and saturation changes. Makes sure the output image is still between 0 and 1.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0, 1).

  Returns:
    image: image which is the same shape as input image.

  Raises:
    ValueError: if color_ordering is not in {0, 1}.
  """
  with tf.name_scope('RandomDistortColor', values=[image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
    else:
      raise ValueError('color_ordering must be in {0, 1}')

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image

############################# black patches ################################

def random_black_patches(image,
                         max_black_patches=10,
                         probability=0.5,
                         size_to_image_ratio=0.1,
                         random_seed=None):
  """Randomly adds some black patches to the image.

  This op adds up to max_black_patches square black patches of a fixed size
  to the image where size is specified via the size_to_image_ratio parameter.

  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_black_patches: number of times that the function tries to add a
                       black box to the image.
    probability: at each try, what is the chance of adding a box.
    size_to_image_ratio: Determines the ratio of the size of the black patches
                         to the size of the image.
                         box_size = size_to_image_ratio *
                                    min(image_width, image_height)
    random_seed: random seed.

  Returns:
    image
  """
  def add_black_patch_to_image(image):
    """Function for adding one patch to the image.

    Args:
      image: image

    Returns:
      image with a randomly added black box
    """
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    box_size = tf.to_int32(
        tf.multiply(
            tf.minimum(tf.to_float(image_height), tf.to_float(image_width)),
            size_to_image_ratio))
    normalized_y_min = tf.random_uniform(
        [], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)
    normalized_x_min = tf.random_uniform(
        [], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)
    y_min = tf.to_int32(normalized_y_min * tf.to_float(image_height))
    x_min = tf.to_int32(normalized_x_min * tf.to_float(image_width))
    black_box = tf.ones([box_size, box_size, 3], dtype=tf.float32)
    mask = 1.0 - tf.image.pad_to_bounding_box(black_box, y_min, x_min,
                                              image_height, image_width)
    image = tf.multiply(image, mask)
    return image

  with tf.name_scope('RandomBlackPatchInImage', values=[image]):
    for _ in range(max_black_patches):
      random_prob = tf.random_uniform(
          [], minval=0.0, maxval=1.0, dtype=tf.float32, seed=random_seed)
      image = tf.cond(
          tf.greater(random_prob, probability), lambda: image,
          lambda: add_black_patch_to_image(image))

    return image


def random_crop_image(image, crop_width_range=(0.8, 1.), crop_height_range=(0.8, 1.),
                      crop_probability=1.0):
  """ randome crop images

   set a random range, this operation will random crop images

   Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    crop_width_range: a tuple of crop width start and end ratio.
    crop_height_range: a tuple of crop height start and end ratio.
    crop_probability: a float probability of crop .

  Returns:
    image: the preprocessed image.
   """
  shp = tf.shape(image)
  crop_height_ratio = tf.random_uniform([], crop_height_range[0], crop_height_range[1])
  crop_height = tf.multiply(tf.cast(shp[0], tf.float32), crop_height_ratio)
  crop_width_ratio = tf.random_uniform([], crop_width_range[0], crop_width_range[1])
  crop_width = tf.multiply(tf.cast(shp[1], tf.float32), crop_width_ratio)
  crop_size = tf.cast(
    tf.round(tf.stack([crop_height, crop_width, 3])), tf.int32)
  crop_size = tf.cond(tf.greater(tf.random_uniform([]), crop_probability),
                      lambda: crop_size, lambda: shp)
  image = tf.random_crop(image, crop_size)
  new_shp = tf.shape(image)
  image = tf.reshape(image, tf.stack([new_shp[0], new_shp[1], 3]))
  return image


def preprocess(image, preprocess_options):
  """Preprocess images and bounding boxes.

  Various types of preprocessing (to be implemented) based on the
  preprocess_options dictionary e.g. "crop image" (affects image and possibly
  boxes), "white balance image" (affects only image), etc. If self._options
  is None, no preprocessing is done.

  Args:
    images: rank 4 float32 tensor contains 1 image -> [1, height, width, 3].
            with pixel values varying between [0, 1]
    preprocess_options: It is a list of tuples, where each tuple contains a
                        function and a dictionary that contains arguments and
                        their values.

  Returns:
    image: the preprocessed image.
  """

  # Preprocess inputs based on preprocess_options
  for option in preprocess_options:
    func, params = option
    image = func(image, **params)

  return image