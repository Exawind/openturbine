cmake_minimum_required (VERSION 3.21 FATAL_ERROR)

if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

project(
  OpenTurbine 
  VERSION 0.0.0
  DESCRIPTION "Wind turbine structural dynamics simulation code"
  LANGUAGES CXX)

include(cmake/PreventInSourceBuilds.cmake)
include(OpenTurbineOptions.cmake)

openturbine_setup_options()

openturbine_global_options()
include(Dependencies.cmake)
openturbine_setup_dependencies()

openturbine_local_options()

set(
  GIT_SHA
  "Unknown"
  CACHE STRING "SHA this build was generated from")
STRING(
  SUBSTRING "${GIT_SHA}"
  0
  8
  GIT_SHORT_SHA
)

add_library(OpenTurbine::openturbine_options ALIAS openturbine_options)
add_library(OpenTurbine::openturbine_warnings ALIAS openturbine_warnings)

add_subdirectory(src)

include(CTest)

if(${OpenTurbine_ENABLE_TESTS})
  add_subdirectory(tests)
endif()
