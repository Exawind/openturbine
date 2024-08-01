function(openturbine_setup_dependencies)
  find_package(KokkosKernels REQUIRED)

  find_package(Amesos2 REQUIRED)

  if(TARGET Kokkos::MKL)
    find_package(MKL REQUIRED)
  endif()

  if(OpenTurbine_ENABLE_VTK)
    find_package(VTK REQUIRED IOXML)
  endif()

  if(OpenTurbine_ENABLE_TESTS)
    find_package(GTest REQUIRED)
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
      set(FS_LIB stdc++fs)
    endif()
  endif()
endfunction()