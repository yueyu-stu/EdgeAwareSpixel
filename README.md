# Edge-aware Superpixel Segmentation with Unsupervised Convolutional Neural Networks

## Environment
Main requirement: Pytorch == 1.6.0  
Docker with nvidia-container-toolkit is recommended.

## Usage
1. Adjust parameters in `utils/config.py`  
   * `IMG_PATH` and `LBL_PATH` should be customized to the dataset folders.  
   * New folders should be created to store generated images and NPY data. If the desired number of superpixels is N, the format is `img/to/path/N/`, where `img/to/path` is a user-defined folder name.
   * `NUM_SPIXELS` inidicates the number of superpixels.

2. Generate superpixels  
Run `python inference.py`

## Acknowledgment
* We measure the results with scipts of [this repo](https://github.com/davidstutz/superpixel-benchmark).
* We utilize codes of Side Window Filter in Pytorch version from [this repo](https://github.com/wang-kangkang/SideWindowFilter-pytorch).
* This work is based on [ss-with-RIM](https://github.com/DensoITLab/ss-with-RIM).

## License
* This work, i.e. EdgeAwareSpixel, is licensed under the MIT License.
* [ss-with-RIM](https://github.com/DensoITLab/ss-with-RIM) is licensed under a customized license.
* License of Side Window Filter in Pytorch version is default.

## Update

This work bases on Information Maximization, so as [ss-with-RIM](https://github.com/DensoITLab/ss-with-RIM). To implement regularization, one could modify Adam optimizer by
* Pytorch 1.6: substitute Adam with AdamW and pass weigh_decay parameter when initializing the optimizer
* Pytorch 1.9: pass weight_decay paramter when initalizing Adam optimizer, since two optimizers realize the strategy of regularization in the same way

For this work, after adding regularization, BR and PR do not change, while ASA becomes slightly higher. We think that
* the edge-aware term could adhere to edges in images strongly
* regularization term could restrain the generation of meaningless superpixels
