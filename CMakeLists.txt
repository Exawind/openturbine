
cmake_minimum_required (VERSION 3.10)
project(OpenTurbine)

set(SPACK_CXX $ENV{SPACK_CXX})
if(SPACK_CXX)
    message("found spack compiler ${SPACK_CXX}")
    set(CMAKE_CXX_COMPILER ${SPACK_CXX} CACHE STRING "the C++ compiler" FORCE)
    set(ENV{CXX} ${SPACK_CXX})
endif()

set(Kokkos_DIR "$ENV{Kokkos_ROOT}" CACHE STRING "Kokkos root directory")
find_package(Kokkos REQUIRED)

add_executable(test_kokkos src/main.cpp)
target_link_libraries(test_kokkos Kokkos::kokkos)
