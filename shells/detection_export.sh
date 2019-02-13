#! /bin/bash

SHELL_DIR=$(cd $(dirname $0); pwd)

PROJECT_DIR=$(dirname ${SHELL_DIR})

DETECTION_DIR="${PROJECT_DIR}/detection"

if [[ $PYTHONPATH != *${DETECTION_DIR}* ]]; then
    export PYTHONPATH=$PYTHONPATH:${DETECTION_DIR}:${DETECTION_DIR}/object_detection/slim
fi

TASK_NAME=$1
CONFIG_PATH=$2

if [[ -z $1 || -z $2 ]]; then
    echo "Usage:"
    echo "bash detection_export.sh task_name config_path"
    echo "    task_name      the name of this task, like cola_cls100_classify/resnet_v1_101"
    echo "    config_path    the detection model pipline config path"
    echo "Please set arguments: task_name and config_path"
    exit
fi

TRAIN_DIR=/home/store-1-img/Logdir/train/${TASK_NAME}
checkpoint=${TRAIN_DIR}/checkpoint
TRAINED_CHECKPOINT_PREFIX=$(cat ${checkpoint} | awk -F\" '{printf "%s\n",$2}' | sed -n "1p")

TRAINED_STEPS=$(echo ${TRAINED_CHECKPOINT_PREFIX} | awk -F "model.ckpt-" '{printf "%s", $2}')

PREFIX=$(echo ${TASK_NAME}_ | tr "/" "_")

python ${DETECTION_DIR}/object_detection/export_inference_graph.py \
  --input_type=image_tensor \
  --pipeline_config_path=${CONFIG_PATH} \
  --trained_checkpoint_prefix=${TRAINED_CHECKPOINT_PREFIX} \
  --output_directory=/home/store-1-img/Logdir/exported/${TASK_NAME}/${TRAINED_STEPS}\
  --tensor_prefix=${PREFIX}

echo "exported model to '/home/store-1-img/Logdir/exported/${TASK_NAME}/${TRAINED_STEPS}'"