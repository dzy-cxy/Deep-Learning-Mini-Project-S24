# Advanced Implementation of ResNet-18 for CIFAR-10 classification

## Requirements
- Python 3.6+
- PyTorch 1.6.0+

## Usage
1. Train

```
mkdir path/to/checkpoint_dir
python train.py --checkpoint_dir path/to/checkpoint_dir

```

2. Test

When your training is done, the model parameter file `path/to/checkpoint_dir/model_final.pth` will be generated.
```
python test.py --params_path path/to/checkpoint_dir/model_final.pth
```

## Note
If you want to specify GPU to use, you should set environment variable `CUDA_VISIBLE_DEVICES=0`, for example.

## References
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep residual learning for image recognition," In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.
