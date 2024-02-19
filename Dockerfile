FROM ubuntu:18.04

MAINTAINER Jesse Morris "jesse.morris@sydney.edu.au"

ENV DEBIAN_FRONTEND=noninteractive

ENV DIRPATH /root/
WORKDIR $DIRPATH

# Add sudo
RUN apt-get update && apt-get install apt-utils sudo -y


RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common nano wget libgflags-dev \
    && add-apt-repository ppa:ubuntu-toolchain-r/test

#Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get update && apt-get install -y git cmake build-essential pkg-config

# Install GCC-9
RUN apt update \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		build-essential \
		gcc-9 \
		g++-9 \
		gcc-9-multilib \
		g++-9-multilib \
		xutils-dev \
		patch \
		git \
		python3 \
		python3-pip \
		libpulse-dev \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 50 \
	&& update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 50


# Install OpenCV for Ubuntu 18.04
RUN apt-get update && apt-get install -y \
      unzip \
      libjpeg-dev libpng-dev libtiff-dev \
      libvtk6-dev \
      libgtk-3-dev \
      libatlas-base-dev gfortran


# Install xvfb to provide a display to container for GUI realted testing.
RUN apt-get update && apt-get install -y xvfb python3-dev python3-setuptools

RUN pip3 install setuptools pre-commit scipy matplotlib argcomplete



RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv && \
      git checkout tags/3.4.0 && \
      mkdir build

RUN git clone https://github.com/opencv/opencv_contrib.git
RUN cd opencv_contrib && \
      git checkout tags/3.4.0

RUN cd opencv/build && \
      cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_opencv_python=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=OFF \
      -DOPENCV_EXTRA_MODULES_PATH=$DIRPATH/opencv_contrib/modules .. && \
      make -j$(nproc) install


# Install GTSAM
RUN apt-get update && apt-get install -y libboost-all-dev libtbb-dev
#ADD https://api.github.com/repos/borglab/gtsam/git/refs/heads/master version.json
RUN git clone --single-branch --branch 4.2a9 https://github.com/borglab/gtsam.git
RUN cd gtsam && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_UNSTABLE=ON -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_ROT3_EXPMAP=ON -DGTSAM_TANGENT_PREINTEGRATION=OFF .. && \
    make -j$(nproc) install


# Install glog, gflags
RUN apt-get update && apt-get install -y libgflags2.2 libgflags-dev libgoogle-glog0v5 libgoogle-glog-dev

# install CSparse
RUN DEBIAN_FRONTEND=noninteractive apt install -y libsuitesparse-dev


SHELL ["/bin/bash", "--login", "-c"]
