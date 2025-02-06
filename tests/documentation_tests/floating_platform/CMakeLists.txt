cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Set C++ standard, prefer 17, no extensions
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to be used")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project declaration
project(
  FloatingPlatform
  VERSION 0.0.0
  DESCRIPTION "A flexible multibody structural dynamics code for wind turbines"
  LANGUAGES CXX
)

find_package(OpenTurbine REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(Amesos2 REQUIRED)
find_package(yaml-cpp REQUIRED)
if(Amesos2_Enable_MPI)
  find_package(MPI REQUIRED)
endif()

add_executable(floating_platform)
target_link_libraries(floating_platform PRIVATE OpenTurbine::openturbine_library)
target_sources(floating_platform PRIVATE floating_platform.cpp)
