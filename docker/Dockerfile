# Ubuntu 18.04 with nvidia-docker2 beta opengl and cuda
# --- How-to ---
# (Build)
# ./build.bash .
# (Run)
#./run.bash dave_nvidia

ARG BASEIMG=nvidia/opengl:1.0-glvnd-devel-ubuntu18.04
FROM $BASEIMG

# Set ROS distribution
ARG DIST=melodic

# Set Gazebo verison
ARG GAZ=gazebo9

# Tools useful during development.
RUN apt-get update \
 && apt-get install -y \
        build-essential \
        cppcheck \
        curl \
        cmake \
        lsb-release \
        gdb \
        git \
        python3-dbg \
        python3-pip \
        python3-venv \
        ruby \
        software-properties-common \
        sudo \
        vim \
        wget \
        libeigen3-dev \
        pkg-config \
        protobuf-compiler \
        nano \
        htop \
        mesa-utils \
        gnupg2 \
        curl \
        ca-certificates \
 && apt-get clean

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y \
    tzdata \
 && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata \
 && apt-get clean

# Get ROS melodic and Gazebo 9.
RUN /bin/sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
 && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
 && /bin/sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list' \
 && wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add - \
 && apt-get update \
 && apt-get install -y \
    python-rosdep \
    qtbase5-dev \
    libgles2-mesa-dev \
    ros-${DIST}-desktop-full \
    ros-${DIST}-velodyne-gazebo-plugins \
    ros-${DIST}-effort-controllers \
    ros-${DIST}-rqt \
    ros-${DIST}-rqt-robot-plugins \
    ros-${DIST}-rqt-common-plugins \
    ros-${DIST}-joy \
    ros-${DIST}-teleop-twist-joy \
    ros-${DIST}-teleop-twist-keyboard \
    ros-${DIST}-teleop-tools \
    ros-${DIST}-joy-teleop \
    ros-${DIST}-key-teleop \
    ros-${DIST}-geographic-info \
    ros-${DIST}-move-base \
    ros-${DIST}-robot-localization \
    ros-${DIST}-robot-state-publisher \
    ros-${DIST}-xacro \
    libignition-math2 \
 && rosdep init \
 && apt-get clean

RUN rosdep update

# --- CUDA --- #
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDA_VERSION 11.1.0
ENV NCCL_VERSION 2.7.8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        nvidia-cuda-toolkit \
        cuda-cudart-11-1=11.1.74-1 \
        cuda-compat-11-1 \
        pkg-config \
        libglvnd-dev libglvnd-dev:i386 \
        libgl1-mesa-dev libgl1-mesa-dev:i386 \
        libegl1-mesa-dev libegl1-mesa-dev:i386 \
        libgles2-mesa-dev libgles2-mesa-dev:i386 \
        cuda-nvml-dev-11-1=11.1.74-1 \
        cuda-command-line-tools-11-1=11.1.0-1 \
        cuda-nvprof-11-1=11.1.69-1 \
        libnpp-dev-11-1=11.1.1.269-1 \
        cuda-libraries-dev-11-1=11.1.0-1 \
        cuda-minimal-build-11-1=11.1.0-1 \
        libcublas-dev-11-1=11.2.1.74-1 \
        libcusparse-11-1=11.2.0.275-1 \
        libcusparse-dev-11-1=11.2.0.275-1 \
    && apt-mark hold libnccl-dev \
    && ln -s cuda-11.1 /usr/local/cuda

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.1 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 brand=tesla,driver>=450,driver<451"

# Set USER and GROUP
ARG USER=developer
ARG GROUP=developer

# Add a user with the same user_id as the user outside the container
# Requires a docker build argument `user_id`.

RUN curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml

RUN addgroup --gid 1000 $USER && \
    adduser --uid 1000 --ingroup $USER --home /home/$USER --shell /bin/sh --disabled-password --gecos "" $USER

RUN adduser $USER sudo \
 && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USER

# Commands below run as the developer user.
USER $USER:$GROUP

# When running a container start in the developer's home folder.
WORKDIR /home/$USER


# Create workspace
RUN mkdir -p uuv_ws/src/

# Source all the needed environment files.
RUN /bin/sh -c 'echo ". /opt/ros/${DIST}/setup.bash" >> ~/.bashrc' \
 && /bin/sh -c 'echo ". /usr/share/gazebo/setup.sh" >> ~/.bashrc'

ENTRYPOINT ["fixuid"]

CMD ["/bin/bash"]
# Customize your image here.


