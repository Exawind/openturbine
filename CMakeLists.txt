
############################ BASE ######################################

cmake_minimum_required (VERSION 3.20 FATAL_ERROR)
project(OpenTurbine CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(openturbine-utils)

########################## OPTIONS #####################################

# General options for the project
option(OTURB_ENABLE_ALL_WARNINGS "Show most warnings for most compilers" ON)
# option(OTURB_ENABLE_CLANG_TIDY "Compile with clang-tidy static analysis" OFF)
# option(OTURB_ENABLE_CPPCHECK "Enable cppcheck static analysis target" OFF)
# option(OTURB_ENABLE_FCOMPARE "Enable building fcompare when not testing" OFF)

# Enabling tests overrides the executable options
# option(OTURB_ENABLE_UNIT_TESTS "Enable unit testing" ON)
option(OTURB_ENABLE_TESTS "Enable testing suite" OFF)
# option(OTURB_SAVE_GOLDS "Provide a directory in which to save golds during testing" OFF)
# option(OTURB_ENABLE_FPE_TRAP_FOR_TESTS "Enable FPE trapping in tests" ON)

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
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.0")
    message(FATAL_ERROR "Your nvcc version is ${CMAKE_CUDA_COMPILER_VERSION} which is unsupported."
      "Please use CUDA toolkit version 10.0 or newer.")
  endif()
endif()

if(OTURB_ENABLE_ROCM)
  find_package(HIP REQUIRED)
  if(NOT DEFINED AMD_ARCH)
    # Set default AMD architectures (based on Frontier)
    set(AMD_ARCH "gfx90a")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(OTURB_ENABLE_FPE_TRAP_FOR_TESTS OFF)
  message(WARNING "Disabling FPE trapping for tests when using AppleClang.")
endif()

############################# OpenTurbine ##############################

generate_version_info()

# General information about machine, compiler, and build type
message(STATUS "OpenTurbine Information:")
message(STATUS "VERSION = ${OTURB_VERSION_TAG}-${OTURB_REPO_DIRTY}")
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

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

init_code_checks()
if(CLANG_TIDY_EXE)
    set_target_properties(
        ${oturb_exe_name} ${oturb_lib_name} # ${oturb_api_lib}
        PROPERTIES CXX_CLANG_TIDY ${CLANG_TIDY_EXE}
    )
endif()

include(set_compile_flags)

# Build OpenTurbine
add_subdirectory(src)

if(OTURB_ENABLE_CUDA)
  set(oturb_targets "${oturb_lib_name};${oturb_exe_name}")
  foreach(tgt IN LISTS oturb_targets)
    set_cuda_build_properties(${tgt})
    # get_target_property(OTURB_SOURCES ${tgt} SOURCES)
    # list(FILTER OTURB_SOURCES INCLUDE REGEX "\\.cpp")
    # set_source_files_properties(${OTURB_SOURCES} PROPERTIES LANGUAGE CUDA)
  endforeach()
endif()

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
      get_target_property(UTEST_SOURCES ${oturb_unit_test_exe_name} SOURCES)
      set_source_files_properties(${UTEST_SOURCES} PROPERTIES LANGUAGE CUDA)
      set_target_properties(${oturb_unit_test_exe_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    endif()
endif()

# add_subdirectory(tools)

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
