function(openturbine_setup_dependencies)
  # Find and link required packages
  find_package(KokkosKernels REQUIRED)
  find_package(Amesos2 REQUIRED)
  find_package(yaml-cpp REQUIRED)

  # Add external project to build the ADI library from OpenFAST
  include(ExternalProject)

  # print CMKE_BINARY_DIR
  message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

  ExternalProject_Add(OpenFAST_ADI
    PREFIX ${CMAKE_BINARY_DIR}/../OpenFAST_ADI
    GIT_REPOSITORY https://github.com/OpenFAST/openfast.git
    GIT_TAG dev
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE FALSE
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -DBUILD_SHARED_LIBS=ON                # Build shared libraries
               -DBUILD_TESTING=OFF                   # Disable tests
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target aerodyn_inflow_c_binding
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE FALSE
  )

  ExternalProject_Get_Property(OpenFAST_ADI SOURCE_DIR)
  message(STATUS "SOURCE_DIR: ${SOURCE_DIR}")
  # Define the OpenFAST_ADI shared library for the aerodyn_inflow_c_binding
  set_target_properties(OpenFAST_ADI PROPERTIES
    IMPORTED_LOCATION "${SOURCE_DIR}/build/modules/aerodyn/aerodyn_inflow_c_binding/libaerodyn_inflow_c_binding.so"
  )


  # Optionally find and link MKL if available
  if(TARGET Kokkos::MKL)
    find_package(MKL REQUIRED)
  endif()

  # Conditionally find and link VTK if enabled
  if(OpenTurbine_ENABLE_VTK)
    find_package(VTK REQUIRED COMPONENTS IOXML)
  endif()

  # Conditionally find and link GTest if testing is enabled
  if(OpenTurbine_ENABLE_TESTS)
    find_package(GTest REQUIRED)
  endif()

  # Link filesystem library for older GCC versions
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
      set(FS_LIB stdc++fs)
    endif()
  endif()
endfunction()
