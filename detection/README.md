## detection

### augment data

past item images to random background images

### convert tfrecord

convert object detection data to tfrecord file

### object detection

train, evaluate, export object detection model

Note: before train, evaluate, export an object detection model, you should cd into this detection folder and
run script ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/object_detection/slim``` in shell