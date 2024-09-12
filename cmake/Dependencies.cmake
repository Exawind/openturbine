function(openturbine_setup_dependencies)
  # Find and link required packages
  find_package(KokkosKernels REQUIRED)
  find_package(Amesos2 REQUIRED)
  find_package(yaml-cpp REQUIRED)

  # Add external project to build the OpenFAST/AerodynInflow(ADI) library
  # include(ExternalProject)
  # ExternalProject_Add(OpenFAST_ADI
  #   PREFIX ${CMAKE_BINARY_DIR}/../OpenFAST_ADI
  #   GIT_REPOSITORY https://github.com/OpenFAST/openfast.git
  #   GIT_TAG dev # Use the branch dev
  #   GIT_SHALLOW TRUE # Only clone the latest commit
  #   GIT_SUBMODULES_RECURSE FALSE # Do not clone submodules
  #   CMAKE_ARGS -DBUILD_TESTING=OFF # Do not build tests
  #   BUILD_IN_SOURCE FALSE # Build in a separate directory
  #   BUILD_COMMAND ${CMAKE_COMMAND} --build . --target aerodyn_inflow_c_binding # Build only the ADI library
  #   INSTALL_COMMAND ""
  # )

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
