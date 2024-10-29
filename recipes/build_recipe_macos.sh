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
install_if_missing cppcheck
install_if_missing yaml-cpp
install_if_missing "trilinos@16.0.0~mpi~epetra"
#install_if_missing llvm # add if clang-tidy is needed
#install_if_missing "vtk~mpi~opengl2" # add if VTK is needed

spack load cppcheck trilinos googletest yaml-cpp #llvm vtk

# Check if FC (Fortran Compiler) is set
if [ -z "${FC}" ]; then
    echo "Error: FC (Fortran Compiler) environment variable is not set."
    echo "Please set FC to your preferred Fortran compiler before running this script."
    echo "For example: export FC=/path/to/your/gfortran"
    exit 1
fi

# Build OpenTurbine with the specified options
mkdir -p build-from-script
cd build-from-script
cmake .. \
  -DOpenTurbine_ENABLE_VTK=ON \
  -DOpenTurbine_BUILD_OPENFAST_ADI=ON \
  -DOpenTurbine_BUILD_ROSCO_CONTROLLER=ON \
  -DCMAKE_BUILD_TYPE="Release"

cmake --build .
