# Use an official Ubuntu as a parent image
FROM ubuntu:latest

# Install required dependencies
RUN apt-get update && \
    apt-get install -y \
    clang-format-14 \
    cmake \
    make \
    g++ \
    gfortran \
    git \
    liblapack-dev \
    liblapacke-dev \
    libgtest-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone Kokkos and build it
RUN git clone -b 3.7.01 https://github.com/kokkos/kokkos && \
    cd kokkos && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# Clone Kokkos-Kernels and build it
RUN git clone -b 3.7.00 https://github.com/kokkos/kokkos-kernels && \
    cd kokkos-kernels && \
    mkdir build && \
    cd build && \
    cmake ../ -DKokkosKernels_ENABLE_TPL_BLAS=ON && \
    make && \
    make install

# Set up GoogleTest
RUN cd /usr/src/gtest && \
    cmake CMakeLists.txt && \
    make && \
    cp lib/*.a /usr/lib && \
    ln -s /usr/lib/libgtest.a /usr/local/lib/libgtest.a && \
    ln -s /usr/lib/libgtest_main.a /usr/local/lib/libgtest_main.a

# Create a workspace directory
WORKDIR /workspace

# Copy your project source code to the container
COPY . .

# Build your project
RUN mkdir build && \
    cd build && \
    cmake .. -DOTURB_ENABLE_TESTS:BOOL=ON && \
    make -j 4
