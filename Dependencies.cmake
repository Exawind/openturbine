function(openturbine_setup_dependencies)
  # Find and link required packages
  find_package(KokkosKernels REQUIRED)
  find_package(Amesos2 REQUIRED)
  find_package(yaml-cpp REQUIRED)

  # Add external project to build the OpenFAST/AerodynInflow(ADI) library
  include(ExternalProject)
  ExternalProject_Add(OpenFAST_ADI
    PREFIX ${CMAKE_BINARY_DIR}/../OpenFAST_ADI
    GIT_REPOSITORY https://github.com/OpenFAST/openfast.git
    GIT_TAG dev # Use the branch dev
    GIT_SHALLOW TRUE # Only clone the latest commit
    GIT_SUBMODULES_RECURSE FALSE # Do not clone submodules
    CMAKE_ARGS -DBUILD_SHARED_LIBS=ON                # Build shared libraries for linking
               -DBUILD_TESTING=OFF                   # Disable tests
    BUILD_IN_SOURCE FALSE # Build in a separate directory
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --target aerodyn_inflow_c_binding # Build only the ADI library
    INSTALL_COMMAND ""
  )

  # Set the location of the ADI library for linking
  ExternalProject_Get_Property(OpenFAST_ADI SOURCE_DIR) # Get the source directory
  message(STATUS "SOURCE_DIR: ${SOURCE_DIR}")
  set_target_properties(OpenFAST_ADI PROPERTIES
    IMPORTED_LOCATION "${SOURCE_DIR}/build/modules/aerodyn/aerodyn_inflow_c_binding/libaerodyn_inflow_c_binding.dylib"
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
