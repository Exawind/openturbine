FROM ubuntu:latest

# Use bash as the default shell
SHELL ["/bin/bash", "-c"]

# Step 1: Install required packages and tools
RUN apt-get update -qq && \
    apt-get install -y \
        build-essential \
        clang-tidy \
        cmake \
        curl \
        g++ \
        gfortran \
        git \
        libcurl4-openssl-dev \
        make \
        python3 \
        python3-pip \
        unzip \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Step 2: Get spack and install necessary libraries using it
RUN git clone https://github.com/spack/spack.git && \
    source spack/share/spack/setup-env.sh && \
    spack compiler find && \
    spack external find && \
    spack mirror add spack-public https://mirror.spack.io && \
    spack install googletest && \
    spack --insecure install cppcheck && \
    spack install yaml-cpp && \
    spack --insecure install trilinos@16.0.0~mpi~epetra && \
    spack --insecure install vtk~mpi~opengl2

# Step 3: Set up the working directory and copy the OpenTurbine repository
WORKDIR /projects/openturbine
COPY . .

# Step 4: Load necessary libraries and build the project
ARG CMAKE_OPTIONS="-DOpenTurbine_ENABLE_SANITIZER_ADDRESS=ON \
    -DOpenTurbine_ENABLE_SANITIZER_LEAK=ON \
    -DOpenTurbine_ENABLE_SANITIZER_UNDEFINED=ON \
    -DOpenTurbine_ENABLE_CPPCHECK=ON \
    -DOpenTurbine_ENABLE_CLANG_TIDY=ON \
    -DOpenTurbine_ENABLE_VTK=ON \
    -DOpenTurbine_BUILD_OPENFAST_ADI=ON \
    -DOpenTurbine_BUILD_ROSCO_CONTROLLER=ON \
    -DCMAKE_BUILD_TYPE=Debug"

ARG N_PARALLEL_JOBS=4

RUN . /spack/share/spack/setup-env.sh && \
    spack load cppcheck googletest trilinos vtk yaml-cpp && \
    mkdir docker-build && \
    cd docker-build && \
    cmake .. ${CMAKE_OPTIONS} && \
    cmake --build . -j ${N_PARALLEL_JOBS} && \
    ctest --output-on-failure -j ${N_PARALLEL_JOBS}
