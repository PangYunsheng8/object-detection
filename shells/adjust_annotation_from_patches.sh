#! /bin/bash

SHELL_DIR=$(cd $(dirname $0); pwd)

PROJECT_DIR=$(dirname ${SHELL_DIR})

TOOLS_DIR="${PROJECT_DIR}/classification/tools"

if [[ -z $1 || -z $2 || -z $3 ]]; then
    echo "Usage:"
    echo "bash detection_eval.sh patches_root, image_folder annotation_folder"
    echo "    patches_root      the folder path of cleaned patches"
    echo "    image_folder      the folder path of source images"
    echo "    annotation_folder the folder path of source annotations "
    echo "Please set arguments: patches_root, image_folder and annotation_folder"
    exit
fi

PATCHES_ROOT=$1
IMAGE_FOLDER=$2
ANNOTATION_FOLDER=$3
if [[ ${ANNOTATION_FOLDER:0-1:1} == '/' ]]
then
  len=`expr ${#ANNOTATION_FOLDER} - 1`
  ANNOTATION_FOLDER=${ANNOTATION_FOLDER:0:$len}
fi

python ${TOOLS_DIR}/adjust_annotation_from_patches.py \
  --patches_root ${PATCHES_ROOT} \
  --annotation_folder ${ANNOTATION_FOLDER} \
  --adjusted_annotation_folder ${ANNOTATION_FOLDER}_adjusted


python ${TOOLS_DIR}/get_object_patches.py \
  --image_folder ${IMAGE_FOLDER} \
  --annotation_folder ${ANNOTATION_FOLDER}_adjusted \
  --output_folder ${ANNOTATION_FOLDER}_adjusted_pathes


python ${TOOLS_DIR}/check_two_folder_be_same.py \
  --folderA ${ANNOTATION_FOLDER}_adjusted_pathes \
  --folderB ${PATCHES_ROOT} \
  --skip_empty

read -p "replace annotations with adjusted annotations: ([y]|n)" replace
if [[ ${replace} == 'y' || ${replace} == '' ]]
then
  cp -r ${ANNOTATION_FOLDER}_adjusted/* ${ANNOTATION_FOLDER}
  rm -r ${ANNOTATION_FOLDER}_adjusted
fi

read -p "update patches: ([y]|n)" update
if [[ ${update} == 'y' || ${update} == '' ]]
then
  DS_FOLDER=$(dirname $(dirname ${ANNOTATION_FOLDER}))
  SUB_FOLDER=${ANNOTATION_FOLDER##*/}
  if [[ -d ${DS_FOLDER}/patches/${SUB_FOLDER} ]]
  then
    rm -r ${DS_FOLDER}/patches/${SUB_FOLDER}
  fi
  mv ${ANNOTATION_FOLDER}_adjusted_pathes ${DS_FOLDER}/patches/${SUB_FOLDER}
else
  rm -r ${ANNOTATION_FOLDER}_adjusted_pathes
fi
