(compiling)=
# Compiling

OpenTurbine is written in C++ and compiles into an both an executable and
a static or shared library. The procedure and considerations for compiling
are described here.

## Dependencies

The third party dependencies used in OpenTurbine are listed below.
These can be installed by any means appropriate to the target
use. For HPC, it is recommended to use vendor specific libraries
when available. For workstations, package managers such as
conda, APT, and Homebrew will provide the easiest experience
for linking and runtime path searches.

- [CMake](<https://cmake.org/>) version 3.20 or higher
- C++ compiler that supports the C++14 standard; 
  - GCC 5 or higher
  - LLVM Clang 7 or higher
  - Intel 2020 (oneAPI) or higher
- [Kokkos](https://github.com/kokkos/kokkos) math portability library (see {ref}`installing-kokkos`)
- OS: OpenTurbine is regularly tested on Linux and macOS
- (Optional) Google Test (gtest) for the test infrastructure (conda-forge)
- (Optional) clang-format for linting (conda-forge or brew)


(installing-kokkos)=
## Installing Kokkos

OpenTurbine relies heavily on Kokkos for portability between systems,
compilers, and hardware types. It is at the core of the this software.
Therefore, the Kokkos library used should be tuned to your specific
use case. See the [Kokkos documentation](https://kokkos.github.io/kokkos-core-wiki/building.html)
for instructions on building that library.

For use in OpenTurbine, the Kokkos library must be available
within a typical search path or you must provide the search
path via the `Kokkos_DIR` environment variable. This will
typically be the install location from your Kokkos build.
It should be set in the shell session where you're compiling
OpenTurbine. See the example below.

```bash
# Configure CMake in OpenTurbine without setting the environment variable
cmake ..

# Results in this error:
# CMake Error at src/CMakeLists.txt:3 (find_package):
#   By not providing "FindKokkos.cmake" in CMAKE_MODULE_PATH this project has
#   asked CMake to find a package configuration file provided by "Kokkos", but
#   CMake did not find one.
# 
#   Could not find a package configuration file provided by "Kokkos" with any
#   of the following names:
# 
#     KokkosConfig.cmake
#     kokkos-config.cmake
# 
#   Add the installation prefix of "Kokkos" to CMAKE_PREFIX_PATH or set
#   "Kokkos_DIR" to a directory containing one of the above files.  If "Kokkos"
#   provides a separate development package or SDK, be sure it has been
#   installed.

# Set the environment variable
export Kokkos_DIR=~/Development/kokkos/install

# Test that is was set correctly
echo $Kokkos_DIR
# Displays:
# ~/Development/kokkos/install

# Reconfigure CMake in OpenTurbine
cmake ..
```

## Build system

The build system is defined entirely within a CMake project.
It is configured via configuration variables that accept either
boolean (`ON`/`OFF`) or string arguments. This is typically
done via the command line interface for CMake:

```bash
cmake .. -DBOOL_FLAG=ON -DSTRING_FLAG="value"
```

See the [CMake documentation](https://cmake.org/documentation/)
for a full reference on CMake. The primary targets available to build
are listed below.

| Target         | Type | Description |
| ----------- | ----- | ----------- |
| openturbine  | Exe | Primary executable including the `main`    |
| openturbine_obj   | Lib | Library including all code used by `main`    |
| openturbine_unit_tests   | Exe |  Unit test driver executable including all tests and links to the Google Test   |

### Architecture options

#### OTURB_ENABLE_OPENMP

   Enable OpenMP threading support for CPU builds. It is not recommended to

   combine this with GPU builds. Default: OFF

#### OTURB_ENABLE_CUDA

   Enable [NVIDIA CUDA GPU](https://developer.nvidia.com/cuda-zone) builds. Default: OFF

#### OTURB_ENABLE_ROCM

   Enable [AMD ROCm GPU](https://rocmdocs.amd.com/en/latest/) builds. Default: OFF

(oturb-enable-dpcpp)=
#### OTURB_ENABLE_DPCPP

   Enable [Intel OneAPI DPC++](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html) builds. Default: OFF

#### OTURB_PRECISION

   Specifies the floating point precision; can be one of "SINGLE" or "DOUBLE". Default: DOUBLE

### Other OpenTurbine specific options

#### OTURB_ENABLE_TESTS

   Adds the testing infrastructure to the build. Default: OFF

#### OTURB_ENABLE_ALL_WARNINGS

   Enable compiler warnings during build. Default: OFF

### General CMake options

#### CMAKE_INSTALL_PREFIX

   The directory where the compiled executables and libraries as well as headers
   are installed. For example, passing
   `-DCMAKE_INSTALL_PREFIX=${HOME}/software` will install the executables in
   `${HOME}/software/bin` when the user executes the `make install` command.

#### CMAKE_BUILD_TYPE

   Controls the optimization levels for compilation. This variable can take the
   following values:


| Value          |  Typical flags |
| ---- | ---- |
| RELEASE |          `-O3 -DNDEBUG` |
| DEBUG |            `-g` |
| RelWithDebInfo |   `-O2 -g` |

   Example: `-DCMAKE_BUILD_TYPE:STRING=RELEASE`

#### CMAKE_CXX_COMPILER

   Set the C++ compiler used for compiling the code.

   For Intel DPC++ builds (see {ref}`oturb-enable-dpcpp`) this should be
   set to `dpcpp`.

#### CMAKE_CXX_FLAGS

   Additional flags to be passed to the C++ compiler during compilation.


## Step by step

### 1. Load or install dependencies

If you are on an HPC system that provides Modules Environment, load the
necessary dependencies. If targeting GPUs, load CUDA
modules.

### 2. Clone the repository

This creates a local copy of the software repository
from GitHub.

```bash
git clone https://github.com/exawind/openturbine.git
```

### 3. Configure and build

This step configures the CMake project and compiles the
code into a "build/" directory within the repository directory.
Be sure to note the location from where the commands are run.
The final step can be run with no additional arguments
to compile all targets or with a specific target. No arguments
instructs to compile all targets.

```bash
mkdir openturbine/build
cd openturbine/build

cmake .. -DOTURB_ENABLE_TESTS:BOOL=ON
make
```

Upon successfully building, two executables should be available
in the build directory:
- `openturbine`: the main driver for the software
- `openturbine_unit_tests`: the driver for the unit tests

### 4. Test your build

Ensure the code is compiled and linked correctly by running the
included tests. These involve low-level unit tests as well
as high-level regression tests. See Testing for more info.

```bash
ctest --output-on-failure
```

## Quirks and issues

Here's a list of known issues or oddities and workarounds.

### GTest from conda-forge

If you've installed Google Test (GTest) with conda, you
must also install Google Mock (GMock) as it is a dependency
but it is not automatically included. The command below
installs both.

```bash
conda install gtest gmock -c conda-forge
```

On Linux with GTest from conda-forge and the GNU
compiler, you may see this error:
`undefined reference to 'std::__throw_bad_array_new_length()@GLIBCXX_3.4.29'`.
In that case, upgrade to GCC 11 (`g++-11`). GTest in conda-forge is linked
to `GLIBCXX_3.4.29` but GCC 10 has `GLIBCXX_3.4.28`. 
Check installed versions with this command:
```bash
strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```
For Ubuntu 20.04, instructions on upgrading GCC are [here](https://lindevs.com/install-gcc-on-ubuntu).

On macOS with GTest from conda-forge, you must include the
GTest directory in the library search path since it will not
be found automatically by rpath. 

```bash
export DYLD_LIBRARY_PATH=~/miniconda3/envs/openturbine/lib  # customize this to your path
```

