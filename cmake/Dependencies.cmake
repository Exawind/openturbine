function(openturbine_setup_dependencies)
  #--------------------------------------------------------------------------
  # Required packages
  #--------------------------------------------------------------------------
  find_package(KokkosKernels REQUIRED)
  find_package(yaml-cpp REQUIRED)
  find_package(NetCDF REQUIRED)
  find_package(LAPACK REQUIRED)

  #--------------------------------------------------------------------------
  # Optional packages
  #--------------------------------------------------------------------------
  #----------------------------------------
  # Sparse Direct Linear Solvers
  #----------------------------------------
  if(OpenTurbine_ENABLE_SUPERLU)
    find_package(superlu REQUIRED)
  endif()

  if(OpenTurbine_ENABLE_SUPERLU_MT)
    find_package(superlu_mt REQUIRED)
  endif()

  if(OpenTurbine_ENABLE_KLU)
    find_package(KLU REQUIRED)
  endif()

  if(OpenTurbine_ENABLE_UMFPACK)
    find_package(UMFPACK REQUIRED)
  endif()

  if(OpenTurbine_ENABLE_MKL)
    find_package(MKL REQUIRED)
  endif()

  if(OpenTurbine_ENABLE_CUSOLVERSP)
    if(NOT DEFINED Kokkos_ENABLE_CUDA)
      message(FATAL_ERROR "When OpenTurbine_ENABLE_CUSOLVERSP is enabled, Kokkos must also be built with CUDA")
    endif()
  endif()

  if(OpenTurbine_ENABLE_CUDSS)
    if(NOT DEFINED Kokkos_ENABLE_CUDA)
      message(FATAL_ERROR "When OpenTurbine_ENABLE_CUDSS is enabled, Kokkos must also be built with CUDA")
    endif()
    find_package(cudss REQUIRED)
  endif()

  #----------------------------------------
  # GTest
  #----------------------------------------
  if(OpenTurbine_ENABLE_TESTS)
    find_package(GTest REQUIRED)
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
  if(OpenTurbine_ENABLE_OPENFAST_ADI)
    find_library(OpenFast_ADI_LIBRARY NAMES aerodyn_inflow_c_binding)
    set(OpenTurbine_ADI_LIBRARY ${OpenFast_ADI_LIBRARY} CACHE PATH "ADI library")
  endif()

  #----------------------------------------
  # ROSCO Controller library
  #----------------------------------------
  if(OpenTurbine_ENABLE_ROSCO_CONTROLLER)
    find_library(Rosco_LIBRARY NAMES discon)
    set(OpenTurbine_ROSCO_LIBRARY ${Rosco_LIBRARY} CACHE PATH "Rosco discon library")
  endif()
endfunction()
