<div align="center">
  <a href="https://robotics.sydney.edu.au/our-research/robotic-perception/">
    <img align="center" src="docs/acfr_rpg_logo.png" width="150" alt="acfr-rpg">
  </a> 
</div>


# Dynamic Slam Coordinates


The offical code used for our paper:
- [Jesse Morris](https://jessemorris.github.io/), Yiduo Wang, Viorela Ila [*The Importance of Coordinate Frames in Dynamic SLAM*](https://arxiv.org/abs/2312.04031).  IEEE Intl. Conf. on Robotics and Automation (ICRA), 2024.

We kindly ask to cite our paper if you find this work useful:

```bibtex

@article{morris2023importance,
  title={The Importance of Coordinate Frames in Dynamic SLAM},
  author={Morris, Jesse and Wang, Yiduo and Ila, Viorela},
  journal={arXiv preprint arXiv:2312.04031},
  year={2023}
}

```
Please view our [project page](https://acfr-rpg.github.io/dynamic_slam_coordinates/
) for the supplementary video and presentation recorded for ICRA 2024.


# 1. Install

Tested on Ubuntu 18.04 and 20.04

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
- Path to the local `dynamic_slam_coords` folder where the code is installed. e.g. if the repo was cloned into `~/code/src/` this argument should be `~/code/src/dynamic_slam_coords`. This is used to mount the code into the container at `/root/dynamic_slam_coordinates`. Mounting the code into the container means the code can be changed locally and built within the docker container. This is the intended workflow.


# 2. Running
Compile the code
```
mkdir build && cmake .. && make
```
Run the code
```
./dsc_app --flagfile=<path to flags file>
```
e.g.
```
./build/dsc_app --flagfile=example/config.flags
```


## Configuration and Graph files

As explained in the paper we evalulate this on sequences from the KITTI dataset. The input data used for these experiments can be found on [google drive](https://drive.google.com/drive/folders/11ZNYnf8G4Zz79aUptc4lKMnIz_6RMQye?usp=sharing). Each sequence is then processed using the front-end described in:

* <b>VDO-SLAM: A Visual Dynamic Object-aware SLAM System</b> <br> 
Jun Zhang\*, Mina Henein\*, Robert Mahony and Viorela Ila. 
<i>	ArXiv:2005.11052</i>.
<a href="https://arxiv.org/abs/2005.11052" target="_blank"><b>[ArXiv/PDF]</b></a>
<a href="https://github.com/halajun/VDO_SLAM" target="_blank"><b>[Code]</b></a>
<a href="https://halajun.github.io/files/zhang20vdoslam.txt" target="_blank"><b>[BibTex]</b></a>

to produce a graph-file describing the the data-assocaition and connections between camera poses, object points and static points. Some modification has been made to the files to include additional information (e.g. object id and frame id per object point) to make the task of evaluation easier, so the output format is not exactly the same as in VDO-SLAM. 

The input datasets still need to be provided to the program (via the `path_to_kitti` flag argument) so we can access the ground truth data for evaluation. The name of the kitti dataset must match the name of the graph-file used as input to the system.


All graph files can be found at [here](https://drive.google.com/drive/folders/1rmUlHzftE9FFYIDS4rrBAsr77WmBaBAl?usp=sharing), where the ones used for this paper come directly from `kitti_graph_initialwithmotion/<kitti dataset>/smooth_robust`.

> Note: The program expects the graph naming format to be `dynamic_slam_graph_after_opt_<kitti dataset>.g2o`. E.g the file for KITI dataset 0018 should be `dynamic_slam_graph_after_opt_0018.g2o`.

> Note: Some graph files have a trailing zero in the prior pose edge (with label EDGE_SE3_PRIOR). This may look like `EDGE_SE3_PRIOR 1 0 ...`, remove the 0, as it will mess with the way the graph files are loaded. This will have to be done manually. 


The config file (that is specified when running the program) specifies some key parameters
- kitti_dataset: The name of the kitti dataset to evaluate. e.g. 0000, 0004, 0018. The program will then form the expected graph file name based on the dataset number as explained above.
- path_to_kitti: Path to the folder containing all the kitti datasets. The folder name of the actual dataset loaded will then be `path_to_kitti/kitti_dataset`
- output_path: Path to where all the output files are saved to.
- graph_file_folder_path: The program will search for the graph files in this folder. 

All other params and settings including which variation of the object-centric formulation to construct and evaluate should also be specified here.