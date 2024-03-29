FROM ubuntu:20.04

ARG NUM_JOBS=14
ARG CMAKE_VERSION=3.23.2
ARG OPENCV_VERSION=4.6.0

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt update && apt install -y \
    build-essential \
    software-properties-common \
    autoconf \
    automake \
    libtool \
    pkg-config \
    ca-certificates \
    wget \
    git \
    git-lfs \
    curl \
    libjpeg-dev \
    libpng-dev \
    language-pack-en \
    locales \
    locales-all \
    libprotobuf-dev \
    protobuf-compiler \
    zlib1g-dev \
    swig \
    vim \
    valgrind \
    libsm6 \
    libxext6 \
    libxrender-dev \
    cmake \
    unzip \
    libssl-dev \
&& apt clean && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm -rf /tmp/*

# Install OpenCV
RUN apt update && apt install -y \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libpng-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev

RUN cd /tmp && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    mkdir -p build && cd build && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DINSTALL_C_EXAMPLES=OFF \
        -DINSTALL_TESTS=OFF \
        ../opencv-${OPENCV_VERSION} && \
    cmake --build . -- -j ${NUM_JOBS} && \
    make install && \
    rm -rf /tmp/*

# Install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.0/onnxruntime-linux-x64-1.12.0.tgz && \
    tar -xvzf onnxruntime-linux-x64-1.12.0.tgz && \
    cp -r onnxruntime-linux-x64-1.12.0/* /usr/ && \
    rm -rf onnxruntime-linux-x64-1.12.0*

# Install POCO
RUN git clone -b master https://github.com/pocoproject/poco.git && \
    cd poco && \
    mkdir cmake-build && cd cmake-build && \
    cmake .. && \
    cmake --build . --config Release -- -j ${NUM_JOBS} && \
    cmake --build . --target install && \
    cd ../../ && \
    rm -rf poco

# Install server
RUN git clone https://github.com/resolator/cpp_classification_server.git /server && \
    cd /server && \
    git-lfs install && \
    git-lfs pull && \
    mkdir build && cd build && \
    cmake .. && \
    cmake --build . --config Release -- -j ${NUM_JOBS}

# Update ldconfig
RUN ldconfig

# Run server
CMD exec /server/build/server_binary --model-path /server/data/resnet18-v1-7.onnx --labels-path /server/data/labels.txt
