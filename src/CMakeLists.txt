# Define the main library target for OpenTurbine
add_library(openturbine_library)
add_library(OpenTurbine::openturbine_library ALIAS openturbine_library)

# Set target properties for openturbine_library
set_target_properties(openturbine_library PROPERTIES
  LINKER_LANGUAGE CXX
  POSITION_INDEPENDENT_CODE ON  # Ensure position-independent code for shared libraries
  VISIBILITY_INLINES_HIDDEN YES # Hide inline function symbols to reduce binary size
)

# Link internal libraries and external dependencies
target_link_libraries(openturbine_library PRIVATE
  openturbine_options
  openturbine_warnings
)

# Link system libraries
target_link_system_libraries(openturbine_library PRIVATE
  KokkosKernels::kokkoskernels
  Amesos2::amesos2
  yaml-cpp
  ${FS_LIB}
)

# Set include directories for the library
target_include_directories(openturbine_library
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>   # Include source directory during build
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>   # Include binary directory during build
)

# Set the C++ standard for the target
target_compile_features(openturbine_library PUBLIC cxx_std_17)

# Define the DISCON shared libraries for the turbine controller
add_library(DISCON SHARED)
set_target_properties(DISCON PROPERTIES PREFIX "" SUFFIX ".dll")

add_library(DISCON_ROTOR_TEST_CONTROLLER SHARED)
set_target_properties(DISCON_ROTOR_TEST_CONTROLLER PROPERTIES PREFIX "" SUFFIX ".dll")

# Add subdirectories for additional components of the library
add_subdirectory(restruct_poc)
add_subdirectory(io)
add_subdirectory(utilities)
