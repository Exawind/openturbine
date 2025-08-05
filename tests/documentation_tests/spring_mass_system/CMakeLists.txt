cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Set C++ standard, prefer 17, no extensions
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to be used")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project declaration
project(
  SpringMassSystem
  VERSION 1.0.0
  DESCRIPTION ""
  LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenTurbine REQUIRED)

add_executable(spring_mass_system)
target_link_libraries(spring_mass_system PRIVATE OpenTurbine::openturbine_library)
target_sources(spring_mass_system PRIVATE spring_mass_system.cpp)
