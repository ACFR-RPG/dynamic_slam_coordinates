# dynamic_slam_coordinates

## Publications


We kindly ask to cite our paper if you find this platform useful:

- Jesse Morris, Yiduo Wang, Viorela Ila [*The Importance of Coordinate Frames in Dynamic SLAM*](https://arxiv.org/abs/2312.04031).  IEEE Intl. Conf. on Robotics and Automation (ICRA), 2024.

```bibtex
@article{morris2023importance,
  title={The Importance of Coordinate Frames in Dynamic SLAM},
  author={Morris, Jesse and Wang, Yiduo and Ila, Viorela},
  journal={arXiv preprint arXiv:2312.04031},
  year={2023}
}
 ```

# 1. Tested on Ubuntu 18.04 and 20.04

## Prerequisites
- [GTSAM](https://github.com/borglab/gtsam) >= 4.2
- [OpenCV](https://github.com/opencv/opencv) >= 3.4
- [Glog](http://rpg.ifi.uzh.ch/docs/glog.html), [Gflags](https://gflags.github.io/gflags/)
- [Gtest](https://github.com/google/googletest/blob/master/googletest/docs/primer.md) (installed automagically).
- gcc >= 9 (tested with 9)

> Note: if you want to avoid building all dependencies yourself, we provide a docker image that will install them for you. All development and testing was done within the image.

## Building docker
We use docker buildkit to build the image
```
docker build -f docker/Dockerfile -t dynamic_slam_coords .
```
This will build a docker image `dynamic_slam_coords` with all the prerequisites but without the source code installed. Instead we will mount the source code directly into the container upon creation for easier modification, via the [creation script](./docker/create_container.sh)

Once the image is built, create the container 
```
./docker/create_container.sh <path to datasets> <output_results_folder> <local dynamic_slam_coordinates folder>
```
The arguments are
- Path to the dataset (on the local machine). This will be mounted in the container at  `/root/data/` for easy access.
- Path the folder (on the local machine) where you want the the results to be saved to.  This will be mounted in the container at  `/root/results` easy access. The output directory can be changed with the config file but this is used to give the program an access point on the local machine
- Path to the local `dynamic_slam_coords` folder where the code is installed. e.g. if the repo was cloned into `~/code/src/` this argument should be `~/code/src/dynamic_slam_coords`. This is used to mount the code into the container at `/root/dynamic_slam_coordinates`. Mounting the code locally into the container means the code can be changed locally and built within the docker container. This is the intended workflow.



## Running
```
./dsc_app --flagfile=<path to flags file>
```
e.g.
```
./build/dsc_app --flagfile=example/config.flags
```

