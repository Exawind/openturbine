function(openturbine_setup_dependencies)
  #--------------------------------------------------------------------------
  # Required packages
  #--------------------------------------------------------------------------
  find_package(KokkosKernels REQUIRED)
  find_package(Amesos2 REQUIRED)
  find_package(yaml-cpp REQUIRED)

  #--------------------------------------------------------------------------
  # Optional packages
  #--------------------------------------------------------------------------
  #----------------------------------------
  # MKL
  #----------------------------------------
  if(TARGET Kokkos::MKL)
    find_package(MKL REQUIRED)
  endif()

  #----------------------------------------
  # MPI
  #----------------------------------------
  if(Amesos2_ENABLE_MPI)
    find_package(MPI REQUIRED)
  endif()

  #----------------------------------------
  # VTK
  #----------------------------------------
  if(OpenTurbine_ENABLE_VTK)
    find_package(VTK REQUIRED COMPONENTS IOXML)
  endif()

  #----------------------------------------
  # GTest
  #----------------------------------------
  if(OpenTurbine_ENABLE_TESTS)
    find_package(GTest REQUIRED)
  endif()

  #----------------------------------------
  # NetCDF and HDF5
  #----------------------------------------
  if(OpenTurbine_ENABLE_NETCDF_OUTPUTS)
    find_package(NetCDF REQUIRED)
    find_package(HDF5 REQUIRED)
  endif()

  #--------------------------------------------------------------------------
  # Compiler-specific settings
  #--------------------------------------------------------------------------
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
      set(FS_LIB stdc++fs)
    endif()
  endif()

  #--------------------------------------------------------------------------
  # External projects
  #--------------------------------------------------------------------------
  #----------------------------------------
  # OpenFAST/AerodynInflow (ADI) library
  #----------------------------------------
  if(OpenTurbine_BUILD_OPENFAST_ADI)
    message(STATUS "Building OpenFAST AerodynInflow (ADI) library")
    include(ExternalProject)
    ExternalProject_Add(OpenFAST_ADI
      PREFIX ${CMAKE_BINARY_DIR}/external
      GIT_REPOSITORY https://github.com/OpenFAST/openfast.git
      GIT_TAG v4.0.0                 # Pin to a specific release
      GIT_SHALLOW TRUE               # Clone only the latest commit
      GIT_SUBMODULES ""              # Skip downloading r-test
      CMAKE_ARGS
        -DBUILD_TESTING=OFF          # Disable testing
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}  # Use the same build type as the main project
      BUILD_IN_SOURCE OFF            # Build in a separate directory for cleaner output
      BINARY_DIR ${CMAKE_BINARY_DIR}/OpenFAST_ADI_build
      # Build only the aerodyn_inflow_c_binding target and do it sequentially (avoid parallel build)
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --target aerodyn_inflow_c_binding -- -j 1
      INSTALL_COMMAND
        # Copy the built library to the tests directory
        ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/OpenFAST_ADI_build/modules/aerodyn/${CMAKE_SHARED_LIBRARY_PREFIX}aerodyn_inflow_c_binding${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${CMAKE_BINARY_DIR}/tests/regression_tests/aerodyn_inflow_c_binding.dll
    )
  endif()

  #----------------------------------------
  # ROSCO Controller library
  #----------------------------------------
  if(OpenTurbine_BUILD_ROSCO_CONTROLLER)
    if (NOT ROSCO_BUILD_TAG)
      set(ROSCO_BUILD_TAG "v2.9.4")
    endif()
    message(STATUS "Building ROSCO Controller library")
    include(ExternalProject)
    ExternalProject_Add(ROSCO_Controller
      PREFIX ${CMAKE_BINARY_DIR}/external
      GIT_REPOSITORY https://github.com/NREL/ROSCO.git
      GIT_TAG ${ROSCO_BUILD_TAG}     # Use tagged release
      GIT_SHALLOW TRUE               # Clone only the latest commit
      BUILD_IN_SOURCE OFF            # Build in a separate directory for cleaner output
      BINARY_DIR ${CMAKE_BINARY_DIR}/ROSCO_build
      SOURCE_SUBDIR rosco/controller
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}  # Use the same build type as the main project
      BUILD_COMMAND ${CMAKE_COMMAND} --build . -- -j 1
      INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/ROSCO_build/${CMAKE_SHARED_LIBRARY_PREFIX}discon${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${CMAKE_BINARY_DIR}/tests/regression_tests/ROSCO.dll
      COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/external/src/ROSCO_Controller/Examples/Test_Cases/NREL-5MW/Cp_Ct_Cq.NREL5MW.txt
        ${CMAKE_BINARY_DIR}/external/src/ROSCO_Controller/Examples/Test_Cases/NREL-5MW/DISCON.IN
        ${CMAKE_BINARY_DIR}/tests/regression_tests
    )
  endif()
endfunction()
