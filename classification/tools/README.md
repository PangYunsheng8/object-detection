## classification tools

### purify annotation

remove invalid bndboxes like corner cordinator is invalid, remove repeated bndboxes,
and remove some class bndboxes.

```
python purify_annotation.py
  --annotation_folder \
  --purify_annotation_folder
```

### cut image to object patches

cut image to object patches base on object annotations, and save patches as object name cluster

```
python get_object_patches.py
  --image_folder image_folder \
  --annotation_folder annotation_folder \
  --output_folder output_folder
```

### adjust annotation from correct classified patches

adjust source annotation from correct classified patches, and save new annotations to adjusted annotation folder.

```
python adjust_annotation_from_patches.py \
  --patches_root patches_root \
  --annotation_folder annotation_folder \
  --adjusted_annotation_folder adjusted_annotation_folder
```

### check two folder

check if two folders are same totally.

```
python check_two_folder_be_same.py
  --folderA folder0 \
  --folderB folder1 \
  --skip_empty
```

### show bndboxes on image

visualize bndboxes on images.

```
python show_bndboxs_on_image.py
  --images_dir directory_of_images \
  --annotations_dir directory_of_annotations \
  --outputs_dir directory_of_outputs
```

### split image dataset

```
python split_image_dataset.py
  --image_dataset dataset_root_dir \
  --output_path output_paths \
  --ratios 0.5,0.3,0.2
```

### analysis detection result

compare ground_truth_annotation_xml with preditcion_annotation_xml calculate recall and precesission 

if you want to draw error info on the image,please call the function draw_error_on_img() manually.
```
python analysis_detction_result.py
  --grt_ann_path_or_dir directory_of_ground_truth_annotation_xmls or single file\
  --prd_ann_path_or_dir directory_of_prediction_annotation_xmls or single file\
  --result_path The path of analysis result. Default: './result.txt'\
  --iou_threshold the box IOU threshold. Default: 0.7
```

