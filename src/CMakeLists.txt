add_library(openturbine_library)
add_library(OpenTurbine::openturbine_library ALIAS openturbine_library)

set_target_properties(openturbine_library PROPERTIES 
  LINKER_LANGUAGE CXX
  POSITION_INDEPENDENT_CODE ON
  VISIBILITY_INLINES_HIDDEN YES)

target_link_libraries(openturbine_library PRIVATE
    openturbine_options
    openturbine_warnings
)

target_link_system_libraries(openturbine_library PRIVATE 
  KokkosKernels::kokkoskernels
  Amesos2::amesos2
  ${FS_LIB}
)

target_include_directories(openturbine_library
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)

target_compile_features(openturbine_library PUBLIC cxx_std_17)

add_library(DISCON SHARED)
set_target_properties(DISCON PROPERTIES PREFIX "" SUFFIX ".dll")

add_library(DISCON_ROTOR_TEST_CONTROLLER SHARED)
set_target_properties(DISCON_ROTOR_TEST_CONTROLLER PROPERTIES PREFIX "" SUFFIX ".dll")

add_subdirectory(restruct_poc)
add_subdirectory(io)
add_subdirectory(utilities)