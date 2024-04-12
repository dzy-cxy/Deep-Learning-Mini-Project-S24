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
The training code execution result is in `checkpoint/loss_log.txt`

2. Test

When your training is done, the model parameter file `path/to/checkpoint_dir/model_final.pth` will be generated.
```
python test.py --params_path path/to/checkpoint_dir/model_final.pth
```

3. Jupyter notebooks for easy visualization and verification

You can run the Jupyter notebook file with clear visualization plots, which also clearly prints the final test accuracy and number of parameters.
For plotting the training loss, we just use the log data in `checkpoint/loss_log.txt`.

## Note
If you want to specify GPU to use, you should set environment variable `CUDA_VISIBLE_DEVICES=0`, for example.

## References
- Krizhevsky, A., Hinton, G., & others. (2009). Learning multiple layers of features from tiny images. Toronto, ON, Canada.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770–778).
- DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552.
- Loshchilov, I., & Hutter, F. (2016). Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983.
