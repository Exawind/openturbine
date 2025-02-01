Compiling
=========

OpenTurbine is developed in C++17 and is designed to be buildable on any
system with a compliant compiler. It utilizes
`Kokkos <https://github.com/kokkos/kokkos>`_ and
`Trilinos <https://github.com/trilinos/Trilinos>`_ to ensure performance
portability, allowing it to run on any platform supported by these projects.
We strive to test OpenTurbine on a wide range of platforms, including Linux
and macOS, although it is not feasible to cover every possible configuration.
This document outlines the build procedure verified to work on Linux (RHEL8).
For additional assistance tailored to your specific setup, please contact the
developers.

Spack Installation
------------------

The easiest way to use OpenTurbine is through the `Spack <https://spack.io/>`_
package manager. Once you have downloaded and set up Spack for your
environment, simply run

.. code-block:: bash

    spack install openturbine

To see the latest list of supported configuration options, check out the
package file or run

.. code-block:: bash

    spack info openturbine

Once it is installed, you can load the OpenTurbine library and its
dependencies into your environment using

.. code-block:: bash

    spack load openturbine

Development using Spack Developer Workflow
------------------------------------------

One easy way to set up a development environment for OpenTurbine is to use
Spack's Developer Workflow. To setup an environment for working on
OpenTurbine, setup Spack and then run the following commands:

.. code-block:: bash

    mkdir openturbine
    cd openturbine
    spack env create -d .
    spack env activate .
    spack add openturbine+tests
    spack install
    spack develop openturbine@main
    spack concretize -f
    spack install

OpenTurbine's source code will now be located in the openturbine folder, but
can be accessed from anywhere by

.. code-block:: bash

    spack cd -c openturbine

After editing the code here, it can be rebuilt by running

.. code-block:: bash

    spack install

To run the tests, first access the build folder through the spack command

.. code-block:: bash

    spack cd -b openturbine

Next, the tests can be run either through ctest or directly from the unit
test or regression test executables

.. code-block:: bash

    ctest
    ./tests/unit_tests/openturbine_unit_tests
    ./tests/regression_tests/openturbine_regression_tests

You can also build OpenTurbine from this folder using standard make
commands.

For more information, please see Spack's documentation:
https://spack-tutorial.readthedocs.io/en/latest/tutorial_developer_workflows.html

Building and Developing in OpenTurbine Directly
-----------------------------------------------

The following sections outline how to build and develop OpenTurbine without
Spack's Developer Workflows. The main complication here is that developers
will have to manage their environment and dependencies manually, which may
be an unnecessary complication or a freeing feature, depending on your
perspective.

Dependencies
------------

Before building OpenTurbine, you'll need the following:

- C++ compiler that supports the C++17 standard
- `CMake <https://cmake.org/>`_: the default build system for C++ projects,
  version 3.21 or later
- `Kokkos <https://github.com/kokkos/kokkos>`_: core programming model for
  performance portability
- `KokkosKernels <https://github.com/kokkos/kokkoskernels>`_: performance
  portable linear algebra library
- `Trilinos <https://github.com/trilinos/Trilinos>`_: primarily for the
  Amesos2 sparse direct linear solver package
- `GoogleTest <https://github.com/google/googletest>`_: unit testing package

Installing Third Party Libraries
--------------------------------

There are several methods to obtain the necessary Third Party Libraries
(TPLs) for building OpenTurbine, however the simplest is to use the
`spack <https://spack.io/>`_ package manager. Spack offers a
comprehensive set of features for development and dependency management. The
following is a quick-start guide for installing and loading the TPLs required
to build OpenTurbine.

Clone the spack repository, load the spack environment, and let spack learn about your system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone git@github.com:spack/spack.git
    source spack/share/spack/setup-env.sh
    spack compiler find
    spack external find

Install GoogleTest
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    spack install googletest

Install Trilinos
~~~~~~~~~~~~~~~~

To build OpenTurbine, Kokkos Kernels must be configured to use the LAPACK
and BLAS TPLs. For GPU builds, Trilinos should be compiled with support for
the Basker linear solver. Additionally, we typically disable EPetra to avoid
deprecation warnings. As of this writing, OpenTurbine is compatible with
Trilinos version 16.0.0, which is the latest and default version in spack.

For a simple serial build

.. code-block:: bash

    spack install trilinos~epetra ^kokkos-kernels+blas+lapack

Trilinos can also be compiled with OpenMP support for parallelism on CPU based machines

.. code-block:: bash

    spack install trilinos~epetra+openmp ^kokkos-kernels+blas+lapack

If building for CUDA platforms, Trilinos must be configured with CUDA support

.. code-block:: bash

    spack install trilinos~epetra+basker+cuda ^kokkos-kernels+blas+lapack

If building for ROCm platforms, Trilinos must be configured with ROCm support

.. code-block:: bash

    spack install trilinos~epetra+basker+rocm ^kokkos-kernels+blas+lapack

Trilinos can be built with or without MPI support.

Load the TPLs into your environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    spack load googletest
    spack load trilinos

Trilinos can be compiled with support for various platforms. While it is
assumed that OpenTurbine will inherit compatibility with these platforms,
they have not been tested at the time of writing.

If you choose not to use Spack, you must manually build all dependencies.
Please ensure that the ``Amesos2_DIR``, ``GTest_DIR``, and ``KokkosKernels_DIR``
environment variables are correctly set for these packages. Alternatively,
make sure that CMake's ``find_package`` utility can locate them.

Building OpenTurbine
--------------------

The following is written assuming the TPLs in hand and the environment
configured as described above.

Clone OpenTurbine and setup a build directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone git@github.com:Exawind/openturbine.git
    cd openturbine
    mkdir build
    cd build

Configure cmake
~~~~~~~~~~~~~~~

For a CPU-based build which includes building unit tests, use

.. code-block:: bash

    cmake ../

If Trilinos was built with CUDA support, you will need to use the nvcc_wrapper
for compilation

.. code-block:: bash

    cmake ../ -DCMAKE_CXX_COMPILER=nvcc_wrapper

If Trilinos was built with ROCm support, you will need to use the hipcc program
for compilation

.. code-block:: bash

    cmake ../ -DCMAKE_CXX_COMPILER=hipcc

Build and Test
~~~~~~~~~~~~~~

Currently, OpenTurbine builds several shared libraries by default. To ensure
that their unit tests pass, these libraries must be copied into the directory
where the tests are executed.

.. code-block:: bash

    make -j
    cp src/*.dll tests/unit_tests/
    ctest --output-on-failure

Once built, the unit test executable can also be run directly from the build
directory

.. code-block:: bash

    ./tests/unit_tests/openturbine_unit_tests

Build Options
-------------

OpenTurbine has several build options which can be set either when running
CMake from the command line or through a GUI such as ccmake.

- ``OpenTurbine_ENABLE_CLANG_TIDY`` enables the Clang-Tidy static analysis tool
- ``OpenTurbine_ENABLE_COVERAGE`` enables code coverage analysis using gcov
- ``OpenTurbine_ENABLE_CPPCHECK`` enables the CppCheck static analysis tool
- ``OpenTurbine_ENABLE_IPO`` enables link time optimization
- ``OpenTurbine_ENABLE_PCH`` builds precompiled headers to potentially decrease
  compilation time
- ``OpenTurbine_ENABLE_SANITIZER_ADDRESS`` enables the address sanitizer runtime
  analysis tool
- ``OpenTurbine_ENABLE_SANITIZER_LEAK`` enables the leak sanitizer runtime
  analysis tool
- ``OpenTurbine_ENABLE_SANITIZER_MEMORY`` enables the memory sanitizer runtime
  analysis tool
- ``OpenTurbine_ENABLE_SANITIZER_THREAD`` enables the thread sanitizer runtime
  analysis tool
- ``OpenTurbine_ENABLE_SANITIZER_UNDEFINED`` enables the undefined behavior
  sanitizer runtime analysis tool
- ``OpenTurbine_ENABLE_TESTS`` builds OpenTurbine's test suite
- ``OpenTurbine_ENABLE_UNITY_BUILD`` uses unity builds to potentially decrease
  compilation time
- ``OpenTurbine_ENABLE_VTK`` builds OpenTurbine with VTK support for
  visualization in tests. Will need the VTK TPL to be properly configured
- ``OpenTurbine_WARNINGS_AS_ERRORS`` treats warnings as errors, including
  warnings from static analysis tools
