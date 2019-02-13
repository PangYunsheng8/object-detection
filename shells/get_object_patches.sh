#! /bin/bash

SHELL_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=$(dirname ${SHELL_DIR})
TOOLS_DIR="${PROJECT_DIR}/classification/tools"

if [[ -z $1 ]]; then
    echo "Usage:"
    echo "bash detection_eval.sh src_folder expand_ratio"
    echo "    src_folder      the folder path of source data contains all images and annotations"
    echo "    expand_ratio    the expand ratio of patches. default is 0.2"
    echo "Please set arguments: src_folder"
    exit
fi

SRC_FOLDER=$1


if [[ ${SRC_FOLDER:0-1:1} == '/' ]]
then
  len=`expr ${#SRC_FOLDER} - 1`
  SRC_FOLDER=${SRC_FOLDER:0:$len}
fi

EXP_RATIO=$2
if [[ -z $2 ]]; then
    EXP_RATIO=0.2
fi

if [[ ${EXP_RATIO} == '0' || ${EXP_RATIO} == '0.' || ${EXP_RATIO} == '0.0' ]]; then
    PATCH_FOLDER=patches
else
    PATCH_FOLDER=patches-${EXP_RATIO}
fi


jpg_num=$(find $SRC_FOLDER -maxdepth 1 -name *.jpg | wc -l)
xml_num=$(find $SRC_FOLDER -maxdepth 1 -name *.xml | wc -l)
if [[ $jpg_num != "0" && $xml_num != "0" ]]
then
  nm=$(basename $SRC_FOLDER)
  SRC_FOLDER=$(dirname $SRC_FOLDER)
  python ${TOOLS_DIR}/get_object_patches.py \
    --image_folder ${SRC_FOLDER}/$nm \
    --output_folder $(dirname $SRC_FOLDER)/${PATCH_FOLDER}/$nm \
    --expand_ratio $EXP_RATIO

else
  oldIFS=$IFS 
  IFS=$'\n'
  
  for nm in $(ls $SRC_FOLDER)
  do
    python ${TOOLS_DIR}/get_object_patches.py \
      --image_folder ${SRC_FOLDER}/$nm \
      --output_folder $(dirname $SRC_FOLDER)/${PATCH_FOLDER}/$nm \
      --expand_ratio $EXP_RATIO
  done
  
fi
