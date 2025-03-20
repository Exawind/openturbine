# -----------------------------------------------------------------------------
# Basic project setup
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Set C++ standard requirements -- prefer 17, no extensions
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to be used")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project declaration
project(
  OpenTurbine
  VERSION 0.1.0
  DESCRIPTION "A flexible multibody structural dynamics code for wind turbines"
  LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Project configuration and options
# -----------------------------------------------------------------------------
# Include custom CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
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

# -----------------------------------------------------------------------------
# Version information
# -----------------------------------------------------------------------------
# Generate a shortened Git SHA for versioning
set(GIT_SHA "Unknown" CACHE STRING "SHA this build was generated from")
string(SUBSTRING "${GIT_SHA}" 0 8 GIT_SHORT_SHA)

# Set OpenTurbine Version
set(OPENTURBINE_VERSION_MAJOR 0 CACHE STRING "major version" FORCE)
set(OPENTURBINE_VERSION_MINOR 1 CACHE STRING "minor version" FORCE)
set(OPENTURBINE_VERSION ${OPENTURBINE_VERSION_MAJOR}.${OPENTURBINE_VERSION_MINOR} CACHE STRING "version" FORCE)

# -----------------------------------------------------------------------------
# Library targets
# -----------------------------------------------------------------------------
# Alias libraries for ease of use in targets
add_library(OpenTurbine::openturbine_options ALIAS openturbine_options)
add_library(OpenTurbine::openturbine_warnings ALIAS openturbine_warnings)

# Add the main source directory
add_subdirectory(src)

# -----------------------------------------------------------------------------
# Installation configuration
# -----------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenTurbineConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/OpenTurbineConfig.cmake"
  INSTALL_DESTINATION lib/cmake/OpenTurbine
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/OpenTurbineConfigVersion.cmake"
  VERSION "${version}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/OpenTurbineConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/OpenTurbineConfigVersion.cmake"
  DESTINATION lib/cmake/OpenTurbine
)

# -----------------------------------------------------------------------------
# Optional components
# -----------------------------------------------------------------------------
# Build tests
include(CTest)
if(OpenTurbine_ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Build documentation
if(OpenTurbine_ENABLE_DOCUMENTATION)
  add_subdirectory(docs)
endif()

configure_file(cmake/OpenTurbine_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/src/OpenTurbine_config.h)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/src/OpenTurbine_config.h
        DESTINATION include/OpenTurbine/
)
