
############################ BASE ######################################
cmake_minimum_required (VERSION 3.20 FATAL_ERROR)
project(OpenTurbine LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(openturbine-utils)

########################## OPTIONS #####################################
# General options for the project
option(OTURB_ENABLE_BASIC_SANITIZERS "Compile with address and undefined behavior sanitizers" OFF)
if(OTURB_ENABLE_BASIC_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

# option(OTURB_ENABLE_CLANG_TIDY "Compile with clang-tidy static analysis" OFF)
# option(OTURB_ENABLE_CPPCHECK "Enable cppcheck static analysis target" OFF)
# option(OTURB_ENABLE_FCOMPARE "Enable building fcompare when not testing" OFF)

# Set RELEASE as default build type if none provided
set(SUPPORTED_BUILD_TYPES Debug Release RelWithDebInfo MinSizeRel)
string(REPLACE ";" " " SUPPORTED_BUILD_TYPES_STRING "${SUPPORTED_BUILD_TYPES}")
if(NOT CMAKE_BUILD_TYPE)
  message(WARNING "No Build type provided - using RELEASE as default")
  set(
    CMAKE_BUILD_TYPE Release CACHE STRING
    "Select the type of build from options: ${SUPPORTED_BUILD_TYPES_STRING}"
    FORCE
  )
endif()

# Make build type case-insensitive and check for supported values
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
string(TOUPPER "${SUPPORTED_BUILD_TYPES}" SUPPORTED_BUILD_TYPES_UPPER)
if(NOT BUILD_TYPE_UPPER IN_LIST SUPPORTED_BUILD_TYPES_UPPER)
  message(WARNING "Build type ${CMAKE_BUILD_TYPE} is NOT supported - reverting to RELEASE")
  set(
    CMAKE_BUILD_TYPE RELEASE CACHE STRING
    "Select the type of build from options: ${SUPPORTED_BUILD_TYPES_STRING}"
    FORCE
  )
endif()

# Add some defs based on the build type
if(BUILD_TYPE_UPPER STREQUAL "RELEASE" OR BUILD_TYPE_UPPER STREQUAL "MINSIZEREL")
  add_definitions(-DRELEASE)
elseif(BUILD_TYPE_UPPER STREQUAL "DEBUG" OR BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
  add_definitions(-DDEBUG)
endif()

# Enabling tests overrides the executable options
option(OTURB_ENABLE_TESTS "Enable testing suite" OFF)

# Options for the executable
option(OTURB_ENABLE_OPENMP "Enable OpenMP" OFF)
option(OTURB_ENABLE_CUDA "Enable CUDA" OFF)
option(OTURB_ENABLE_ROCM "Enable ROCm/HIP" OFF)
option(OTURB_ENABLE_DPCPP "Enable Intel OneAPI DPC++" OFF)
set(OTURB_PRECISION "DOUBLE" CACHE STRING "Floating point precision SINGLE or DOUBLE")

# Options for C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(OTURB_ENABLE_CUDA)
else()
  add_compile_options(-Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic -Werror)
endif()

# If CUDA support is enabled, ensure the CUDA language is available
if(OTURB_ENABLE_CUDA)
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.0")
    message(FATAL_ERROR "Your nvcc version is ${CMAKE_CUDA_COMPILER_VERSION} which is unsupported."
      "Please use CUDA toolkit version 10.0 or newer.")
  endif()
endif()

# If ROCm support is enabled, find the required HIP package and set the default AMD GPU architecture
if(OTURB_ENABLE_ROCM)
  find_package(HIP REQUIRED)
  if(NOT DEFINED AMD_ARCH)
    # Set default AMD architectures (based on Frontier)
    set(AMD_ARCH "gfx90a")
  endif()
endif()

# If the compiler is AppleClang, disable the FPE (Floating Point Exception) trapping for tests
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(OTURB_ENABLE_FPE_TRAP_FOR_TESTS OFF)
  message(WARNING "Disabling FPE trapping for tests when using AppleClang.")
endif()

# Enable VTK output for visualization as an option
option(OTURB_ENABLE_VTK "Enable VTK output" OFF)
if(OTURB_ENABLE_VTK)
  find_package(VTK REQUIRED IOXML)
  add_compile_definitions(OTURB_ENABLE_VTK)
endif()

############################# OpenTurbine ##############################
# General information about machine, compiler, and build type
generate_version_info()
message(STATUS "OpenTurbine Information:")
message(STATUS "VERSION = ${OTURB_VERSION_TAG}-${OTURB_REPO_DIRTY}")
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE = ${BUILD_TYPE_UPPER}")

include(set_rpath)

# Create target names
set(oturb_lib_name "openturbine_obj")
set(oturb_exe_name "openturbine")
set(oturb_unit_test_exe_name "${oturb_exe_name}_unit_tests")
set(oturb_api_lib "openturbine_api")

# Create main target executable
add_executable(${oturb_exe_name})
add_library(${oturb_lib_name} OBJECT)
# add_library(${oturb_api_lib})

# Set clang-tidy as the code analysis tool if available
init_code_checks()
if(CLANG_TIDY_EXE)
  set_target_properties(
    ${oturb_exe_name} ${oturb_lib_name} # ${oturb_api_lib}
    PROPERTIES CXX_CLANG_TIDY ${CLANG_TIDY_EXE}
  )
endif()

# Build OpenTurbine
add_subdirectory(src)

# If CUDA support is enabled, set CUDA build properties specified targets
if(OTURB_ENABLE_CUDA)
  set(oturb_targets "${oturb_lib_name};${oturb_exe_name}")
  foreach(tgt IN LISTS oturb_targets)
    set_cuda_build_properties(${tgt})
    # get_target_property(OTURB_SOURCES ${tgt} SOURCES)
    # list(FILTER OTURB_SOURCES INCLUDE REGEX "\\.cpp")
    # set_source_files_properties(${OTURB_SOURCES} PROPERTIES LANGUAGE CUDA)
  endforeach()
endif()

# Set some options for testing
if(OTURB_ENABLE_UNIT_TESTS OR OTURB_ENABLE_TESTS)
  add_executable(${oturb_unit_test_exe_name})
  if(CLANG_TIDY_EXE)
      set_target_properties(
          ${oturb_unit_test_exe_name}
          PROPERTIES CXX_CLANG_TIDY ${CLANG_TIDY_EXE}
      )
  endif()
  add_subdirectory("tests/unit_tests")
  set_cuda_build_properties(${oturb_unit_test_exe_name})
  if(OTURB_ENABLE_CUDA)
    # get_target_property(UTEST_SOURCES ${oturb_unit_test_exe_name} SOURCES)
    # set_source_files_properties(${UTEST_SOURCES} PROPERTIES LANGUAGE CUDA)
    # set_target_properties(${oturb_unit_test_exe_name} PROPERTIES
    #   CUDA_SEPARABLE_COMPILATION ON)
  endif()
  include(GoogleTest)
  gtest_discover_tests(${oturb_unit_test_exe_name} PROPERTIES DISCOVERY_TIMEOUT 60000)
endif()

if(OTURB_ENABLE_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

# Define what we want to be installed during a make install
install(
  TARGETS ${oturb_exe_name} ${oturb_lib_name} # ${oturb_api_lib}
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

if(OTURB_ENABLE_UNIT_TESTS OR OTURB_ENABLE_TESTS)
  install(
    TARGETS ${oturb_unit_test_exe_name}
    RUNTIME DESTINATION bin
  )
endif()

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/src
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.H"
)

install(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

configure_package_config_file(
  cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)

############################# Shared Libraries ############################
# Build DISCON.cpp into a shared library
add_library(DISCON SHARED "${PROJECT_SOURCE_DIR}/src/utilities/controllers/discon.cpp")
set_target_properties(DISCON PROPERTIES PREFIX "" SUFFIX ".dll")

# Build DISCON_ROTOR_TEST_CONTROLLER.cpp into a shared library if tests are enabled
if(OTURB_ENABLE_TESTS)
  add_library(DISCON_ROTOR_TEST_CONTROLLER SHARED "${PROJECT_SOURCE_DIR}/src/utilities/controllers/discon_rotor_test_controller.cpp")
  set_target_properties(DISCON_ROTOR_TEST_CONTROLLER PROPERTIES PREFIX "" SUFFIX ".dll")
endif()
