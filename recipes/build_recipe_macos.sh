#!/bin/bash

# -----------------------------------------------------------------------------
# Run this from the root of the OpenTurbine repository with the following arguments:
# ./recipes/build_recipe_macos.sh <path_to_openturbine_root> <path_to_spack_root>
# -----------------------------------------------------------------------------

# Exit on error
set -e

# Clone OpenTurbine repository if not present in the provided path
openturbine_path=$(dirname "$1") # Get the directory of the provided path
if [ ! -d "$openturbine_path/openturbine" ]; then
    git clone --recursive https://github.com/Exawind/openturbine.git $openturbine_path/openturbine
fi
cd $openturbine_path/openturbine

# Install Spack if not present in the provided path
spack_path=$(dirname "$2") # Get the directory of the provided path
if [ ! -d "$spack_path/spack" ]; then
    git clone https://github.com/spack/spack.git $spack_path/spack
fi
source $spack_path/spack/share/spack/setup-env.sh
spack compiler find
spack external find

# Install required packages using Spack if not present
install_if_missing() {
    if ! spack find -l "$1" | grep -q "$1"; then
        echo "Installing $1..."
        spack install "$1"
    else
        echo "$1 is already installed."
    fi
}

install_if_missing googletest
install_if_missing llvm
install_if_missing cppcheck
install_if_missing yaml-cpp
install_if_missing "vtk~mpi~opengl2"
install_if_missing "trilinos@16.0.0~mpi~epetra"

# Load required packages
spack load llvm
spack load cppcheck
spack load trilinos
spack load googletest
spack load yaml-cpp
spack load vtk

# Find gfortran compiler
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Please install it first."
    exit 1
fi
export FC=$(find /opt/homebrew/ -name gfortran | tr ' ' '\n' | grep "/gcc/.*gfortran")

# Build OpenTurbine with following options - modify as needed
BUILD_TYPE="Release" # Release, Debug, RelWithDebInfo
CXX_COMPILER="clang++" # g++, clang++
BUILD_EXTERNAL="all" # all, none

mkdir -p build
cd build
cmake .. \
  -DOpenTurbine_ENABLE_SANITIZER_ADDRESS=$([[ "$BUILD_EXTERNAL" == "none" ]] && echo "ON" || echo "OFF") \
  -DOpenTurbine_ENABLE_SANITIZER_UNDEFINED=$([[ "$BUILD_EXTERNAL" == "none" ]] && echo "ON" || echo "OFF") \
  -DOpenTurbine_ENABLE_CPPCHECK=ON \
  -DOpenTurbine_ENABLE_CLANG_TIDY=ON \
  -DOpenTurbine_ENABLE_VTK=ON \
  -DOpenTurbine_BUILD_OPENFAST_ADI=$([[ "$BUILD_EXTERNAL" == "all" ]] && echo "ON" || echo "OFF") \
  -DOpenTurbine_BUILD_ROSCO_CONTROLLER=$([[ "$BUILD_EXTERNAL" == "all" ]] && echo "ON" || echo "OFF") \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

cmake --build .

# Run tests
ctest --output-on-failure
