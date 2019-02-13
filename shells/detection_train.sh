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
    echo "bash detection_train.sh task_name config_path"
    echo "    task_name      the name of this task, like cola_cls100_classify/resnet_v1_101"
    echo "    config_path    the detection model pipline config path"
    echo "Please set arguments: task_name and config_path"
    exit
fi

python ${DETECTION_DIR}/object_detection/train.py \
  --logtostderr \
  --train_dir=/home/store-1-img/Logdir/train/${TASK_NAME} \
  --pipeline_config_path=${CONFIG_PATH}