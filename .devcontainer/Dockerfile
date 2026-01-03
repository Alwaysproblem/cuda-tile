FROM alwaysproblem/fastdev-u2204:nv13.1.0

ARG UID=1000
ARG GID=1000

RUN apt update -y && \
    apt install -y \
    python3 python3-dev python3-setuptools python3-pip \
    libtinfo-dev zlib1g-dev \
    build-essential cmake ninja-build

RUN apt update -y && apt install -yq software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt update -yq \
    && apt install -yq gcc-13 g++-13 gdb \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && apt clean -y && rm -rf /var/lib/apt/lists/*

# RUN apt install linux-headers-$(uname -r) && \
#     wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb 
    
# ENV NVIDIA_DRIVER_VERSION=590.48.01
# ENV arch=x86_64
# ENV distro=ubuntu2204
# ENV version=$NVIDIA_DRIVER_VERSION

RUN git config --global --add safe.directory '*' && \
    /root/.local/bin/setup_new_user ${UID} ${GID} && \
    python3 -m pip install numpy pre-commit compdb
