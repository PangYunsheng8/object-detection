# 渠道监控Poc流程

## 数据

1. 数据标注平台[http://label.ainnovation.com/manage/index.html]
2. 导出大图用于检测模型训练，也可用于分类，需要先用`tools/get_object_patches.py`剪切出小图，导出小图用于分类模型训练，建议在标注任务较少时导出大图，这样可以按具体的标注任务导出数据；从平台导出的数据可以直接导出标注或者标注带图片，如果只有标注文件，需要通过标注文件中的图片url去下载图片，下载图片可以用`tools/download_images.py` 。

##　训练及导出

### 分类模型训练

1. 先将数据整理成如下样式的json文件，

```json
{
    "paths": ["p1", "p2],
    "classes": ["c1", "c2"],
    "display_names": ["nm1", "nm2"]
}
```

2. 数据打包成tfrecord

   ```shell
   python classification/convert_to_tfrecord.py --data_info_file file_path \
   ddd
   ```

3. 配置pipline config 文件

   将config模板文件`classification/samples/ebest_carlsberg_resnet_v2_101.config`拷贝到自定义位置，配置config文件中的相应参数，模板文件如下:

   ```protobuf
   spec {
     model_name: "resnet_v2_101"    # 模型
     num_classes: 109    # 类别数量
     dataset_dir: "/home/store-1-img/Datasets/ebest_carlsberg_new/tfrecords/this_one/"   #数据地址
     logdir: "/home/store-1-img/Logdir/train/carlsberg/features"   # 训练log地址
     moving_average_decay: 0.9999
   }
   
   train_config {
      num_clones: 3    # 占用GPU数量
      batch_size: 256    # 根据GPU显存设置batch_size大小
      fine_tune_checkpoint: "/home/store-1-img/PretrainedModels/classification/tensorflow/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt"
      checkpoint_exclude_scopes: "resnet_v2_101/logits"
      # freeze_variables: "resnet_v2_101/block[1234]"
      optimizer {
        adam_optimizer {
          learning_rate {
            exponential_decay_learning_rate {
              initial_learning_rate: 0.002
              decay_steps: 4000000
              decay_factor: 0.95
              staircase: true
            }
          }
        }
      }
      classification_loss {
        softmax_cross_entropy {
        }
      }
      data_augmentation_options {
        normalize_image {
          original_minval: 0
          original_maxval: 255
          target_minval: 0
          target_maxval: 1
        }
      }
      data_augmentation_options {
        random_crop_image {
           width_range {
             start: 0.9
             end: 1.0
           }
           height_range {
             start: 0.9
             end: 1.0
           }
           crop_probability: 1.0
        }
      }
      data_augmentation_options {
        random_horizontal_flip {}
      }
      data_augmentation_options {
        random_vertical_flip {}
      }
      data_augmentation_options {
        random_rotation90 {}
      }
      data_augmentation_options {
        random_pixel_value_scale {}
      }
      # data_augmentation_options {
      #   random_rgb_to_gray {}
      # }
      # data_augmentation_options {
      #   random_distort_color {}
      # }
      data_augmentation_options {
        resize_image {
          new_height: 224
          new_width: 224
        }
      }
    }
   
   eval_config {
     batch_size: 128
     data_preprocess_options {
       normalize_image {}
     }
     data_preprocess_options {
       resize_image {
         new_height: 224
         new_width: 224
       }
     }
   }
   ```


4. 开始训练模型

   ```shell
   python classification/train.py --logtostderr --pipeline_config_path xxx
   ```


5. 测试模型

   ```shell
   python classification/eval.py --logtostderr --pipeline_config_path xxx
   ```

6. 导出模型

   ```shell
   python classification/export.py --logtostderr --pipeline_config_path xxx --export_dir xxx --tensor_prefix xxx
   ```


### 检测模型训练

1. 数据准备

   从标注平台导出大图数据，利用`tools/purify_annotation.py`简单清洗一下标注结果，`detetion/convert_tfrecord/create_od_tfrecord.py`将检测数据打包成tfrecord，用于模型训练。

2. 配置pipeline config文件

   根据任务需要对pipeline config文件进行配置，config文件模板如下：

   ```protobuf
   # Faster R-CNN with Inception v2, configuration for MSCOCO Dataset.
   # Users should configure the fine_tune_checkpoint field in the train config as
   # well as the label_map_path and input_path fields in the train_input_reader and
   # eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
   # should be configured.
   
   
   model {
     faster_rcnn {
       num_classes: 7  # TO_BE_CONFIGURED
       image_resizer {
         keep_aspect_ratio_resizer {
           min_dimension: 640
           max_dimension: 1024
         }
       }
       feature_extractor {
         type: 'faster_rcnn_inception_v2'
         first_stage_features_stride: 16
       }
       first_stage_anchor_generator {
         grid_anchor_generator {
           scales: [0.1, 0.25, 0.5, 1.0, 2.0]
           aspect_ratios: [0.25, 0.333, 0.5, 1.0, 2.0, 3.0, 4.0]
           height_stride: 16
           width_stride: 16
         }
       }
       first_stage_box_predictor_conv_hyperparams {
         op: CONV
         regularizer {
           l2_regularizer {
             weight: 0.0
           }
         }
         initializer {
           truncated_normal_initializer {
             stddev: 0.01
           }
         }
       }
       first_stage_nms_score_threshold: 0.0
       first_stage_nms_iou_threshold: 0.7
       first_stage_max_proposals: 1000
       first_stage_localization_loss_weight: 2.0
       first_stage_objectness_loss_weight: 1.0
       initial_crop_size: 14
       maxpool_kernel_size: 2
       maxpool_stride: 2
       second_stage_box_predictor {
         mask_rcnn_box_predictor {
           use_dropout: false
           dropout_keep_probability: 1.0
           fc_hyperparams {
             op: FC
             regularizer {
               l2_regularizer {
                 weight: 0.0
               }
             }
             initializer {
               variance_scaling_initializer {
                 factor: 1.0
                 uniform: true
                 mode: FAN_AVG
               }
             }
           }
         }
       }
       second_stage_post_processing {
         batch_non_max_suppression {
           score_threshold: 0.0
           iou_threshold: 0.6
           max_detections_per_class: 500
           max_total_detections: 500
         }
         score_converter: SOFTMAX
       }
       second_stage_localization_loss_weight: 2.0
       second_stage_classification_loss_weight: 1.0
     }
   }
   
   train_config: {
     batch_size: 1  # TO_BE_CONFIGURED
     optimizer {
       momentum_optimizer: {
         learning_rate: {
           manual_step_learning_rate {
             initial_learning_rate: 0.0002
             schedule {
               step: 0
               learning_rate: .0002
             }
             schedule {
               step: 50000
               learning_rate: .0002
             }
             schedule {
               step: 300000
               learning_rate: .00004
             }
             schedule {
               step: 500000
               learning_rate: .00002
             }
             schedule {
               step: 1000000
               learning_rate: .00001
             }
           }
         }
         momentum_optimizer_value: 0.9
       }
       use_moving_average: false
     }
     gradient_clipping_by_norm: 10.0
     fine_tune_checkpoint: "/home/zhaojianghua/workspace/cola/checkpoints/faster_rcnn_inception_v2_coco_2017_11_08/model.ckpt"
     from_detection_checkpoint: true
     # Note: The below line limits the training process to 200K steps, which we
     # empirically found to be sufficient enough to train the COCO dataset. This
     # effectively bypasses the learning rate schedule (the learning rate will
     # never decay). Remove the below line to train indefinitely.
     num_steps: 2000000
     data_augmentation_options {
       random_horizontal_flip {
       }
     }
   }
   train_input_reader: {
     tf_record_input_reader {
       input_path: "/home/store-1-img/zhaojianghua/workspace/all_category_detection/data/tfrecords/detection-pzxzgxdzbzhztz/carsberg_quechao_kangshifu_cola_dection.tfrecord-00[0123][0123456789]-of-0050"  # TO_BE_CONFIGURED
       input_path: "/home/store-1-img/zhaojianghua/workspace/all_category_detection/data/tfrecords/detection-pzxzgxdzbzhztz/carsberg_quechao_kangshifu_cola_dection.tfrecord-004[01234]-of-0050"  # TO_BE_CONFIGURED
     }
     label_map_path: "/home/store-1-img/zhaojianghua/workspace/all_category_detection/data/tfrecords/detection-pzxzgxdzbzhztz/label_map.pbtxt"  # TO_BE_CONFIGURED
   }
   
   eval_config: {
     num_examples: 12000
     num_visualizations: 50
     eval_interval_secs: 1800
     # Note: The below line limits the evaluation process to 10 evaluations.
     # Remove the below line to evaluate indefinitely.
     # max_evals: 10
   }
   
   eval_input_reader: {
     tf_record_input_reader {
       input_path: "/home/store-1-img/zhaojianghua/workspace/all_category_detection/data/tfrecords/detection-pzxzgxdzbzhztz/carsberg_quechao_kangshifu_cola_dection.tfrecord-004[56789]-of-0050"
     }  # TO_BE_CONFIGURED
     label_map_path: "/home/store-1-img/zhaojianghua/workspace/all_category_detection/data/tfrecords/detection-pzxzgxdzbzhztz/label_map.pbtxt"  # TO_BE_CONFIGURED
     shuffle: false
     num_readers: 1
   }
   
   ```

3. 训练模型

   可以直接执行如下命令训练模型：

   ```shell
   export PYTHONPATH=PYTHONPATH:`pwd`/detection
   CUDA_VISIBLE_DEVICES=0 python detection/object_detection/train.py --logtostderr --logdir dir/of/tensorflow/trian/log --pipeline_config_path path/of/pipeline/config
   ```

   也可以调用写好的shell脚步运行，建议用shell脚步运行

   `CUDA_VISIBLE_DEVICES=0 bash shells/detection_train.sh task_name pipeline_config_path`

   `task_name`为训练任务名，建议以如下格式命名：{项目名}/{任务名}/{模型名}_{其他细致描述}，最总训练的log会被保存到`/home/store-1-img/Logdir/trian/${task_name}`。

4. 模型测试

   可以直接执行如下命令测试模型：

   ```shell
   export PYTHONPATH=PYTHONPATH:`pwd`/detection
   CUDA_VISIBLE_DEVICES=0 python detection/object_detection/eval.py --logtostderr --logdir dir/of/tensorflow/trian/log --pipeline_config_path path/of/pipeline/config
   ```

   也可以调用写好的shell脚步运行，建议用shell脚步运行

   `CUDA_VISIBLE_DEVICES=0 bash shells/detection_eval.sh task_name pipeline_config_path`

   `task_name`为训练任务名，建议以如下格式命名：{项目名}/{任务名}/{模型名}_{其他细致描述}，最总训练的log会被保存到`/home/store-1-img/Logdir/eval/${task_name}`。

5. 模型导出

   可以直接执行如下命令测试模型：

   ```shell
   export PYTHONPATH=PYTHONPATH:`pwd`/detection
   CUDA_VISIBLE_DEVICES=0 python detection/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path path/of/pipeline/config --trained_checkpoint_prefix path/of/export/checkpoint --output_directory output/directory
   ```

   也可以调用写好的shell脚步运行，建议用shell脚步运行

   `CUDA_VISIBLE_DEVICES=0 bash shells/detection_export.sh task_name pipeline_config_path`

   `task_name`为训练任务名，建议以如下格式命名：{项目名}/{任务名}/{模型名}_{其他细致描述}，最总训练的log会被保存到`/home/store-1-img/Logdir/exported/${task_name}/${checkpoint_step}`。

### 检测分类模型合并

1. 训练好的大类的检测模型
2. 训练好的细类的检测模型
3. 准备大类与细类的映射表
4. 合并模型