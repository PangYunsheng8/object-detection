# Pre-trained models

We have downloaded some classic models' pre-trained checkpoints. You can use them to initial your model for fast convergence and better performance.

You can make a soft link of pre-trained model to your user directory to ease the using of pre-trained models.

```
	ln -s /home/store-1-img/PretrainedModels/object_detection/tensorflow/ checkpoints
```

## How to get pre-trained models

We usually put pre-trained models in our server. Contact IT department for access permission and carefully download them to your checkpoints folder.

The root directory of pre-trained models is ``` /home/store-1-img/PretrainedModels/ ``` on server ```10.18.103.201```, ```10.18.103.205```, ```10.18.103.208```. For each pre-trained model, the path format is ``{task_class}/{framework}/{model_name}_{task_name}_{datetime}``. task_class like 'classification' or 'object_detection'. framework is tensorflow, pytorch or other DL framework. ```/home/store-1-img/PretrainedModels/object_detection/tensorflow/faster_rcnn_resnet50_coco_2018_01_28``` is an example.

For this project, we have put our pre-trained models in ``` /home/store-1-img/PretrainedModels/object_detection/tensorflow/ ``` on server ```10.18.103.208``` and ```10.18.103.205```. You can download it via scp after you got permission from IT department.

## Share your pre-trained models

You can share your trained models. Pick up your trained model and put it to  pre-trained models directory. Make sure the directory path of your model is abey the name rule. Name rule is ``{task_class}/{framework}/user_{model_name}_{task_name}_{datetime}``
