# detection classification API


## save merged model of detection model and classification model

```
python save_det_cls_model.py \
  --detection_model_path detection_model_path \
  --classify_model_path classify_model_path \
  --export_dir export_dir \
  --version 0
```

## two mode of api: Local and Remote

### Local mode: inference in current thread

### Remote: inference on server

- First, install tensorflow serving from [github](https://github.com/tensorflow/serving).
Use bazel to build tensorflow_model_server, add option `--config=cuda` if you want to use GPU.

```
$ bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-O3 --config=cuda tensorflow_serving/...
```
You may encounter compilation some errors about crosstool and nccl when compiling with cuda.  Usually you can fix it by following these steps:
```
1. The crosstool in `tools/bazel.rc` is invalid (AFAIK). change `@org_tensorflow//third_party/gpus/crosstool` to `@local_config_cuda//crosstool:toolchain`.
2. The `cuda_configure` repository rule will fail (haven't looked in to why exactly), but essentially an `bazel clean --expunge && export TF_NEED_CUDA=1` will fix this.
3. Then, run `bazel query 'kind(rule, @local_config_cuda//...)'` again and all is well (for me at least); the cuda tool chain should be created in `$(bazel info output_base)/external/local_config_cuda/cuda`.
```
If bazel complains about `external/nccl_archive/src/nccl.h: No such file or directory` try solve it by these steps:
```
$ git clone https://github.com/NVIDIA/nccl.git
$ cd nccl/
$ make CUDA_HOME=/usr/local/cuda

$ sudo make install
$ sudo mkdir -p /usr/local/include/external/nccl_archive/src
$ sudo ln -s /usr/local/include/nccl.h /usr/local/include/external/nccl_archive/src/nccl.h
$ sudo mkdir -p /usr/local/include/third_party/nccl
$ sudo ln -s /usr/local/include/nccl.h /usr/local/include/third_party/nccl/nccl.h

```
- Second, edit model config file, example as follow:

```
# model config list
model_config_list {
  config {
    # Model name
    name: "mnist"
    # Directory of multi version saved models
    base_path: "absolute/directory/of/saved/models"
    # Model platform
    model_platform: "tensorflow"
  }
}

```

- Then, start tensorflow_model_server

```
tensorflow_model_server --port=9000 --model_config_file=path/to/model/config
```

### API

#### detection only

**inputs:** ndarray-uint8 of image data with shape [1, width, height, 3]

**outputs:** dict. like:
```
{
  "boxes": ndarray-float32 [1, num, 4],
  "scores": ndarray-float32 [num,],
  "classes": ndarray-unicode [num,]
}
```

#### classify only

**inputs:** ndarray-uint8 of image data with shape [1, width, height, 3]

**outputs:** dict. like:
```
{
  "scores": ndarray-float32 [1,],
  "classes": ndarray-unicode [1,]
}
```

#### detection and classify

**inputs:** ndarray-uint8 of image data with shape [1, width, height, 3]

**outputs:** dict. like:
```
{
  "boxes": ndarray-float32 [1, num, 4],
  "scores": ndarray-float32 [num,],
  "classes": ndarray-unicode [num,],
}
```


## do inference

```
python inference.py \
  --mod mod \
  --model_path model_path \
  --label_map_path label_map_path \
  --image_path_or_dir image_path_or_dir \
  --output_dir output_dir
```
