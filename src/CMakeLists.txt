#--------------------------------------------------------------------------
# Define the main library target
#--------------------------------------------------------------------------
add_library(openturbine_library)
add_library(OpenTurbine::openturbine_library ALIAS openturbine_library)

#----------------------------------------
# Set target properties
#----------------------------------------
set_target_properties(openturbine_library PROPERTIES
  LINKER_LANGUAGE CXX
  POSITION_INDEPENDENT_CODE ON  # Ensure position-independent code for shared libraries
  VISIBILITY_INLINES_HIDDEN YES # Hide inline function symbols to reduce binary size
)

#----------------------------------------
# Link internal libraries and dependencies
#----------------------------------------
target_link_libraries(openturbine_library PRIVATE
  openturbine_options
  openturbine_warnings
)

#--------------------------------------------------------------------------
# Link system libraries
#--------------------------------------------------------------------------
#----------------------------------------
# yaml-cpp
#----------------------------------------
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_TARGET yaml-cpp)
endif()

#----------------------------------------
# NetCDF
#----------------------------------------
if(TARGET netCDF::netcdf)
  set(NETCDF_TARGET netCDF::netcdf)
elseif(TARGET NetCDF::NetCDF)
  set(NETCDF_TARGET NetCDF::NetCDF)
elseif(TARGET netcdf)
  set(NETCDF_TARGET netcdf)
else()
  message(FATAL_ERROR "NetCDF library not found")
endif()

#----------------------------------------
# VTK
#----------------------------------------
if(OpenTurbine_ENABLE_VTK)
  target_link_system_libraries(openturbine_library PRIVATE VTK::IOXML)
  target_compile_definitions(openturbine_library PRIVATE OpenTurbine_ENABLE_VTK)
endif()

#--------------------------------------------------------------------------
# Include directories for system libraries
#--------------------------------------------------------------------------
target_link_system_libraries(openturbine_library PUBLIC
  KokkosKernels::kokkoskernels
  Amesos2::amesos2
  ${YAML_CPP_TARGET}
  ${FS_LIB}
  ${NETCDF_TARGET}
)

#----------------------------------------
# Link MPI if it is defined by TPLs
#----------------------------------------
if(TARGET MPI::MPI_CXX)
  target_link_system_libraries(openturbine_library PUBLIC
    MPI::MPI_CXX
  )
endif()

#--------------------------------------------------------------------------
# Include shared libraries
#--------------------------------------------------------------------------
include(../cmake/SharedLibraries.cmake)

#----------------------------------------
# Define shared libraries for controllers
#----------------------------------------
add_shared_library(DISCON)
add_shared_library(DISCON_ROTOR_TEST_CONTROLLER)

#----------------------------------------
# Set include directories
#----------------------------------------
target_include_directories(openturbine_library
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include/OpenTurbine>
)

# Set the C++ standard for the target
target_compile_features(openturbine_library PUBLIC cxx_std_17)

#----------------------------------------
# Add subdirectories
#----------------------------------------
add_subdirectory(constraints)
add_subdirectory(dof_management)
add_subdirectory(elements)
add_subdirectory(interfaces)
add_subdirectory(math)
add_subdirectory(model)
add_subdirectory(solver)
add_subdirectory(state)
add_subdirectory(step)
add_subdirectory(system)
add_subdirectory(utilities)
add_subdirectory(vendor)

#----------------------------------------
# Install targets and export configuration
#----------------------------------------
install(
  TARGETS openturbine_library openturbine_options openturbine_warnings
  EXPORT OpenTurbineTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${PACKAGE_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT OpenTurbineTargets
  FILE OpenTurbineTargets.cmake
  NAMESPACE OpenTurbine::
  DESTINATION lib/cmake/OpenTurbine
)

install(FILES
  types.hpp
  DESTINATION include/OpenTurbine/
)
