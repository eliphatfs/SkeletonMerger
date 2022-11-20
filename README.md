# Skeleton Merger
Skeleton Merger, an Unsupervised Aligned Keypoint Detector.
The paper is available at [https://arxiv.org/abs/2103.10814](https://arxiv.org/abs/2103.10814).

[<img src="_readme_resources/intropic.jpg" width="50%" alt="Intro pic" />](_readme_resources/intropic.jpg)

**Update Aug. 6th:** The point cloud visualizer is now released! See [https://github.com/eliphatfs/PointCloudVisualizer](https://github.com/eliphatfs/PointCloudVisualizer).

## A map of the repository
+ The `merger/pointnetpp` folder contains the [Pytorch Implementation of PointNet and PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) repository with some minor changes. It is adapted to make compatible relative imports.
+ The `merger/composed_chamfer.py` file contains an efficient implementation of proposed Composite Chamfer Distance (CCD).
+ The `merger/data_flower.py` file is for data loading and preprocessing.
+ The `merger/merger_net.py` file contains the `Skeleton Merger` implementation.
+ The root folder contains several scripts for training and testing.

You can find a pre-trained model on chairs from ShapeNetCore [here](https://github.com/eliphatfs/SkeletonMerger/issues/8). Notice that axis order (e.g., gravity axis may be either y or z) and scaling may vary between datasets, so it is recommended to train a model locally from scratch if you need to use Skeleton Merger. It's fast! Skeleton Merger usually gives reasonable results within 5-10 epochs, which only takes minutes on ShapeNetCore-scale datasets with a GTX 1080. (For full power of the model you still need to train for 50-100 epochs and do some epoch selection by validation error or by the downstream task.)

## Usage of script files
Usage of the script files, together with a brief description of data format, are available through the `-h` command line option.

## Dataset
The ShapeNetCore.v2 dataset used in the paper is available from the [Point Cloud Datasets](https://github.com/AnTao97/PointCloudDatasets) repository.
