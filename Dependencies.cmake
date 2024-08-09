function(openturbine_setup_dependencies)
  # Find and link required packages
  find_package(KokkosKernels REQUIRED)
  find_package(Amesos2 REQUIRED)

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
