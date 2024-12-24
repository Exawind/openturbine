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
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_TARGET yaml-cpp)
endif()

target_link_system_libraries(openturbine_library PUBLIC
  KokkosKernels::kokkoskernels
  Amesos2::amesos2
  ${YAML_CPP_TARGET}
  ${FS_LIB}
)

# Link MPI if it is defined by TPLs
if(TARGET MPI::MPI_CXX)
  target_link_system_libraries(openturbine_library PUBLIC
    MPI::MPI_CXX
  )
endif()

# Include shared libraries for the OpenTurbine library
include(../cmake/SharedLibraries.cmake)

# Define the DISCON shared libraries for the turbine controller
add_shared_library(DISCON)
add_shared_library(DISCON_ROTOR_TEST_CONTROLLER)

# Set include directories for the library
target_include_directories(openturbine_library
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>   # Include source directory during build
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>   # Include binary directory during build
)

# Set the C++ standard for the target
target_compile_features(openturbine_library PUBLIC cxx_std_17)

set(openturbine_public_headers "types.hpp" "constraints/calculate_constraint_force.hpp")

# Add subdirectories for additional components of the library
add_subdirectory(constraints)
add_subdirectory(elements)
add_subdirectory(math)
add_subdirectory(model)
add_subdirectory(solver)
add_subdirectory(state)
add_subdirectory(step)
add_subdirectory(system)
add_subdirectory(utilities)
add_subdirectory(vendor)

set_target_properties(openturbine_library PROPERTIES PUBLIC_HEADER "${openturbine_public_headers}")

install(
  TARGETS openturbine_library
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${PACKAGE_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTAL_INCLUDEDIR})
