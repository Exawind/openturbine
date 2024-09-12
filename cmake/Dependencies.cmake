function(openturbine_setup_dependencies)
  # Find and link required packages
  find_package(KokkosKernels REQUIRED)
  find_package(Amesos2 REQUIRED)
  find_package(yaml-cpp REQUIRED)

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

  # Conditionally add external project to build the OpenFAST/AerodynInflow(ADI) library
  if(OpenTurbine_BUILD_OPENFAST_ADI)
    message(STATUS "Building OpenFAST AerodynInflow (ADI) library")
    include(ExternalProject)
    ExternalProject_Add(OpenFAST_ADI
      PREFIX ${CMAKE_BINARY_DIR}/external/OpenFAST_ADI
      GIT_REPOSITORY https://github.com/OpenFAST/openfast.git
      GIT_TAG dev                    # Use the "dev" branch
      GIT_SHALLOW TRUE               # Clone only the latest commit
      GIT_SUBMODULES_RECURSE OFF     # Avoid unnecessary submodule cloning
      CMAKE_ARGS
        -DBUILD_TESTING=OFF          # Disable testing
      BUILD_IN_SOURCE OFF            # Build in a separate directory for cleaner output
      BINARY_DIR ${CMAKE_BINARY_DIR}/OpenFAST_ADI_build
      # Build only the aerodyn_inflow_c_binding taget and do it sequentially (avoid parallel build)
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --target aerodyn_inflow_c_binding -- -j 1
      INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/OpenFAST_ADI_build/modules/aerodyn/libaerodyn_inflow_c_binding${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${CMAKE_BINARY_DIR}         # Copy the library to the binary directory
    )
  endif()

endfunction()
