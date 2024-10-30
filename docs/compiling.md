# Compiling

OpenTurbine is written in C++17 and should be buildable on all systems with a compliant compiler.
Because it leverages Kokkos and Trilinos for performance portability, OpenTurbine is expected to run anywhere that those projects support.
Every effort is made to test on a variety of platforms, including both Linux and MacOS, but it is unlikely that we routinely cover all possibilities.
This page documents the build proceedure as known to work on Linux (RHEL8).  
Please reach out to the developers if additional guidance is needed for your particular situation.

## Dependencies

Before building OpenTurbine, you'll need the following:

- C++ compiler that supports the C++17 standard
- [CMake](<https://cmake.org/>) the default build system for C++ projects, version 3.21+
- [Kokkos](https://github.com/kokkos/kokkos) core programming model for performance portability
- [KokkosKernels](https://github.com/kokkos/kokkoskernels) performance portable linear algebra library
- [Trilinos](https://github.com/trilinos/Trilinos) primarily for the Amesos2 sparse direct linear solver package
- [GoogleTest](https://github.com/google/googletest) unit testing package

## Installing Third Party Librariess

While there are many ways to get the required Third Party Libraries (TPLs) for building, the easiest is the use the [spack](https://github.com/spack/spack) package manager.
Spack provides a rich featureset for development and dependency management.
The following should be considered a quick-start guide for installing and loading the TPLs you'll need for building OpenTurbine.

### Clone the spack repository, load the spack environment, and let spack learn about your system
```bash
git clone git@github.com:spack/spack.git
source spack/share/spack/setup-env.sh
spack compiler find
spack external find
```

### Install GoogleTest
```bash
spack install googletest
```

### Install Trilinos.

For building OpenTurbine, Trilinos must be configured without MPI and with the Basker solver enabled.
We also commonly disable EPetra explicitly to prevent deprication warnings.
At the time of this writing, OpenTurbine is known to work with Trilinos version 16.0.0 - the latest and default version in spack 

For a simple serial build
```bash
spack install trilinos~mpi~epetra+basker
```

Trilinos can also be compiled with OpenMP support for parallelism on CPU based machines
```bash
spack install trilinos~mpi~epetra+basker+openmp
```

If building for CUDA platforms, Trilinos must be configured with CUDA support
```bash
spack install trilinos~mpi~epetra+basker+cuda+cuda_rdc
```

If building for ROCm platforms, Trilinos must be configured with ROCm support
```bash
spack install trilinos~mpi~epetra+basker+rocm
```

### Load the TPLs into your environment
```bash
spack load googletest
spack load trilinos
```

Trilinos can also be compiled with support for other platforms.
It is assumed that OpenTurbine will inherit compatibility with them, but they have not been tested at the time of writing.

For those that choose not to use spack, you must build all of the dependencies manually.  
You will have to ensure the the `Amesos2_DIR`, `GTest_DIR`, and `KokkosKernels_DIR` environment variables are properly set for those packages, or otherwise make sure that cmake's `find_package` utility will be able to find them.  

## Building OpenTurbine

The following is written assuming the TPLs in hand and the environment configured as described above.

### Clone OpenTurbine and setup a build directory
```bash
git clone git@github.com:Exawind/openturbine.git
cd openturbine
mkdir build
cd build
```

### Configure cmake

For a CPU-based build which includes building unit tests, use
```bash
cmake ../
```

If Trilinos was built with CUDA support, you will need to use the nvcc_wrapper for compilation
```bash
cmake ../ -DCMAKE_CXX_COMPILER=nvcc_wrapper
```

If Trilinos was built with ROCm support, you will need to use the hipcc program for compilation
```bash
cmake ../ -DCMAKE_CXX_COMPILER=hipcc
```

### Build and Test
At this time, OpenTurbine builds several shared libraries by default.  
In order for thier unit tests to pass, they will have to be copied into the directory where your tests are run.
```bash
make -j
cp src/*.dll tests/unit_tests/
ctest --output-on-failure
```

Once built, the unit test executable can also be run directly from the build directory
```bash
cp src/*.dll ./
./tests/unit_tests/openturbine_unit_tests
```

### Build Options

OpenTurbine has several build options which can be set either when running cmake from the commandline or through a GUI such as ccmake.

- [OpenTurbine_ENABLE_CLANG_TIDY] enables the Clang-Tidy static analysis tool
- [OpenTurbine_ENABLE_COVERAGE] enables code coverage analysis using gcov
- [OpenTurbine_ENABLE_CPPCHECK] enables the CppCheck static analysis tool
- [OpenTurbine_ENABLE_IPO] enables link time optimization
- [OpenTurbine_ENABLE_PCH] builds precompiled headers to potentially decrease compilation time
- [OpenTurbine_ENABLE_SANITIZER_ADDRESS] enables the address sanitizer runtime analysis tool
- [OpenTurbine_ENABLE_SANITIZER_LEAK] enables the leak sanitizer runtime analysis tool
- [OpenTurbine_ENABLE_SANITIZER_MEMORY] enables the memory sanitizer runtime analysis tool
- [OpenTurbine_ENABLE_SANITIZER_THREAD] enables the thread sanitizer runtime analysis tool
- [OpenTurbine_ENABLE_SANITIZER_UNDEFINED] enables the undefined behavior sanitizer runtime analysis tool
- [OpenTurbine_ENABLE_TESTS] builds OpenTurbine's test suite
- [OpenTurbine_ENABLE_UNITY_BUILD] uses unity builds to potentially decrease compilation time
- [OpenTurbine_ENABLE_VTK] builds OpenTurbine with VTK support for visualization in tests.  Will need the VTK TPL to be properly configured
- [OpenTurbine_WARNINGS_AS_ERRORS] treats warnings as errors, including warnings from static analysis tools
