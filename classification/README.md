# Image Classification model library

This library contains scripts that will allow you to train models from scratch or fine-tune them from pre-trained network weights. It also contains code for building tfrecord dataset.

## overview

The organization is inspired by the [tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection), an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models. Another helps come from [tensorflow slim classification lib](https://github.com/tensorflow/models/tree/master/research/slim), which containing the implementation of popular architectures (ResNet, Inception and VGG). Hence, this repository is separated in three main part:

* core:  implementations of some independent core components, including model deployment  and preprocessor.
* builders: builder for model, dataset, preprocessor, etc.
* nets: classification model implementation forked from slim library directly.

Also, [google protobuf](https://developers.google.com/protocol-buffers/)  is adopted to configure the model spec, training and evaluating process. A brief intro will be given below, and you could read [the definition](./protos) by yourself.

## work flow to train a classification model

### warm up

To begin, compile the `Protobuf` libraries.

```sh
# From ebest_cola/
protoc classification/protos/*.proto --python_out=.
```

Then, expose this library to python.

```sh
# From ebest_cola/
export PYTHONPATH=$PYTHONPATH:`pwd`
```

### prepare the dataset

It's assumed that a `JSON` file containing information of the source image paths and corresponding labels will be given. The` JSON` format is as following:

```json
{  
    "classes":["foo", "bar"],
    "paths":[".../ex_1.jpg",".../ex_1.png"],
    "display_names":["biubiu", "xiuxiu"]
}
```

Select the directory where you want to build the dataset, number of shards the training set as well as validation set contains and the split ratio of training set, execute

```sh
# From ebest_cola/
SOURCE_INFO_PATH=... # the JSON file path.
DATASET_DIR=... # irectory to store the built dataset.
NUM_SHARDS=50 # a separated process is employed to build one shard, appropriate number of shards could save you some time.
TRAIN_SPLIT_RATIO=0.8 # choose whatever you want, but reasonably between 0.7~0.9.
python classification/convert_cls_tfrecords.py \
      --source_info_path=${SOURCE_INFO_PATH} \
      --dataset_dir=${DATASET_DIR} \
      --num_shards=${NUM_SHARDS} \
      --train_split_ratio=${TRAIN_SPLIT_RATIO}
```

The directory structure of the built dataset will be as following:

```
+dataset
  -label_map.pbtxt
  -display_name_map.json
  -dataset_stats.json
  +train
      -train-00001-of-00050.tfrecord
      ...
  +validation
      -validation-00001-of-00050.tfrecord
      ...
```

### train a classification model

As mentioned above,  we adopt `Protobuf`  to configure the model spec, training and evaluating process. There are some [samples](./samples) you can use.

#### configure spec

Spec refers to some common configuration options between training and evaluation, such as:

| field       | type   | description                                                  |
| :---------- | ------ | ------------------------------------------------------------ |
| model_name  | string | classification model name.                                   |
| num_classes | int    | Number of classes to predict.                                |
| dataset_dir | string | dataset directory which contains train and validation split. |
| logdir      | string | Directory where train and eval logs are written to.          |

#### configure training process

Training configuration table is a little complex at first look, but you could leave most of the options as it is. The options you would specify may be:

| field                     | type   | description                                       |
| ------------------------- | ------ | ------------------------------------------------- |
| num_clones                | int    | Number of model clones to deploy.                 |
| batch_size                | int    | The number of samples in each batch.              |
| fine_tune_checkpoint      | string | The path to a checkpoint from which to fine-tune. |
| checkpoint_exclude_scopes | string | Scopes excluded when restoring.                   |
| data_augmentation_options | object | Data augmentation options.                        |
| optimizer                 | object | Optimizer used to train the classification model. |

##### data augmentation options

`data_augmentation_options`  supported now includes:

| option                   | description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| normalize_image          | Normalizes pixel values in an image.                         |
| random_horizontal_flip   | Randomly horizontally flips the image and detections 50% of the time. |
| random_vertical_flip     | Randomly vertically flips the image and detections 50% of the time. |
| random_rotation90        | Randomly rotates the image and detections by 90 degrees counter-clockwise 50% of the time. |
| random_pixel_value_scale | Randomly scales the values of all pixels in the image by some constant value, then clip the value to a range between [0, 1.0]. |
| random_rgb_to_gray       | Randomly convert entire image to grey scale.                 |
| random_adjust_brightness | Randomly changes image brightness.                           |
| random_adjust_contrast   | Randomly scales contract.                                    |
| random_adjust_hue        | Randomly alters hue.                                         |
| random_adjust_saturation | Randomly changes saturation.                                 |
| random_distort_color     | Performs a random color distortion.                          |
| random_black_patches     | Randomly adds black square patches to an image.              |
| resize_image             | Resizes images.                                              |

Combining `random_horizontal_flip`,  `random_vertical_flip`,  and `random_rotation90` produces an output with a uniform distribution of the eight possible 90 degree rotation / reflection combinations.

`random_distort_color`  itself is a combination of `random_adjust_brightness`, `random_adjust_contrast`, `random_adjust_hue`, and `random_adjust_saturation`.

The `data_augmentation_options` should start by `normalize_image` and end with `resize_image`.

##### optimizer

`optimizer` supported now:

- [rms_prop_optimizer](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer)
- [momentum_optimizer](https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer)
- [adam_optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer)
- [adadelta_optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
- [ftrl_optimizer](https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer)
- [sgd](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)

##### learning rate

`learning_rate` comes up as a parameter of optimizer. `learning_rate`  supported now:

- constant_learning_rate
- [exponential_decay_learning_rate](https://www.tensorflow.org/versions/master/api_docs/python/train/decaying_the_learning_rate#exponential_decay)
- [polynomial_decay_earning_rate](https://www.tensorflow.org/versions/master/api_docs/python/train/decaying_the_learning_rate#polynomial_decay)

#### configure evaluating process

The configuration options you may tune:

| field                   | type   | description                                            |
| ----------------------- | ------ | ------------------------------------------------------ |
| batch_size              | int    | The number of samples in each batch.                   |
| data_preprocess_options | object | Data preprocess options.                               |
| max_num_batches         | int    | Max number of batches to evaluate, by default use all. |

`data_preprocess_options` is of the same type as `data_augmentation_options` in training configuration,  just choose `normalize_image` and `resize_image` options for evaluating.

#### kick off training

Now it's time to train, simply open the terminal and run the command below.

```sh
# From ebest_cola/
python classification/train.py \
      --pipeline_config_path=...
```

Press `return` and that's all. Usually you want to evaluate the model at the same time, open another terminal and run the command below.

```sh
# From ebest_cola/
python classification/eval.py \
      --pipeline_config_path=...
```

To visualize the losses and other metrics during training, you can use [TensorBoard](https://github.com/tensorflow/tensorboard) by running the command below.

```sh
LOGDIR=... # LOGDIR was configured in spec.
tensorboard --logdir=${LOGDIR}
```

#### export the model

When finished training, export the model for inference by running the command below.

```sh
# From ebest_cola/
python classification/export.py \
      --pipeline_config_path=... \
      --export_dir=... 
```

## to do

* transfer from slim to tensorflow high level api.