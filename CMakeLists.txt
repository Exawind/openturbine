cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Set C++ standard, prefer 17, no extensions
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to be used")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project declaration
project(
  OpenTurbine
  VERSION 0.0.0
  DESCRIPTION "A flexible multibody structural dynamics code for wind turbines"
  LANGUAGES CXX
)

# Include custom CMake modules
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/OpenTurbineOptions.cmake)

# Setup project-specific options
openturbine_setup_options()
openturbine_global_options()

# Include and setup dependencies
include(cmake/Dependencies.cmake)
openturbine_setup_dependencies()

# Setup local project options
openturbine_local_options()

# Generate a shortened Git SHA for versioning
set(GIT_SHA "Unknown" CACHE STRING "SHA this build was generated from")
string(SUBSTRING "${GIT_SHA}" 0 8 GIT_SHORT_SHA)

# Alias libraries for ease of use in targets
add_library(OpenTurbine::openturbine_options ALIAS openturbine_options)
add_library(OpenTurbine::openturbine_warnings ALIAS openturbine_warnings)

# Add the main source directory
add_subdirectory(src)

# Enable testing if requested
include(CTest)
if(OpenTurbine_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
