
set(Kokkos_DIR "$ENV{Kokkos_ROOT}" CACHE STRING "Kokkos root directory")
find_package(KokkosKernels REQUIRED)
if(TARGET Kokkos::MKL)
  find_package(MKL REQUIRED)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    set(FS_LIB stdc++fs)
  endif()
endif()

target_sources(${oturb_exe_name} PRIVATE main.cpp)
set_target_properties(${oturb_exe_name} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${oturb_exe_name} PRIVATE
    Kokkos::kokkoskernels
    ${FS_LIB}
)

set_target_properties(${oturb_lib_name} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${oturb_lib_name} PRIVATE
    Kokkos::kokkoskernels
    ${FS_LIB}
)

target_include_directories(${oturb_exe_name} PRIVATE ${PROJECT_BINARY_DIR})
target_include_directories(${oturb_exe_name}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

# OpenTurbine lib include directories
target_include_directories(${oturb_lib_name}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Get all source files
# add_subdirectory(gebt_poc)
# add_subdirectory(gen_alpha_poc)
add_subdirectory(restruct_poc)
add_subdirectory(io)
add_subdirectory(utilities)

# Generate OpenTurbine version header
configure_file("${CMAKE_SOURCE_DIR}/cmake/OpenTurbineVersion.H.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenTurbineVersion.H" @ONLY
)

target_link_libraries(${oturb_exe_name} PRIVATE ${oturb_lib_name})

# Fetch dylib
# TODO This should be done via find_package to make it a pre-install dependency
include(FetchContent)
FetchContent_Declare(
  dylib
  GIT_REPOSITORY  "https://github.com/martin-olivier/dylib"
  GIT_TAG         "v2.2.1"
)
FetchContent_MakeAvailable(dylib)

# Build DISCON.c into a shared library
set(SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/utilities/controllers/DISCON.c")
add_library(DISCON SHARED ${SOURCE_PATH})
set_target_properties(DISCON PROPERTIES PREFIX "" SUFFIX ".dll")
add_library(dynamic_lib SHARED ${SOURCE_PATH})

target_link_libraries(${oturb_exe_name} PRIVATE DISCON)

# Set -fpic options
set_target_properties(${oturb_lib_name}
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

file(GLOB_RECURSE openturbine_header_list ${CMAKE_CURRENT_SOURCE_DIR} "*.H")
add_custom_target(openturbine_headers SOURCES ${openturbine_header_list})

# Add CMake-generated headers to installation
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/OpenTurbineVersion.H
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
