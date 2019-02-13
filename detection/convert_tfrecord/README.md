## convert tfrecord

convert detection dataset to tfrecord

```
python create_od_tfrecord.py --output_path output_path \
    --dataset_dir dataset_dir \
    --image_folder Images \
    --annotation_folder Annotations
```



```
python create_od_tfrecord_mask.py --output_path output_path \
    --dataset_dir dataset_dir \
    --image_folder Images \
    --annotation_folder Annotations \
    --mask_folder Mask
```