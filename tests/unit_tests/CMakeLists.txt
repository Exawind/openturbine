# Add the executable target for unit tests
add_executable(openturbine_unit_tests)

# Add subdirectory for additional components (if required)
add_subdirectory(restruct_poc)

# Specify the source files for the unit tests
target_sources(
  openturbine_unit_tests
  PRIVATE
  utest_main.cpp
)

# Link internal libraries to the unit test executable
target_link_libraries(
  openturbine_unit_tests
  PRIVATE
  openturbine_warnings
  openturbine_options
  OpenTurbine::openturbine_library
)

# Link system libraries and dependencies
target_link_system_libraries(
  openturbine_unit_tests
  PRIVATE
  KokkosKernels::kokkoskernels
  Amesos2::amesos2
  yaml-cpp
  GTest::gtest
  GTest::gtest_main
)

# Conditional linking for VTK if enabled
if (OTURB_ENABLE_VTK)
  target_link_system_libraries(
    openturbine_unit_tests
    PRIVATE
    VTK::IOXML
  )
endif()

# Include GoogleTest module and discover tests
include(GoogleTest)
gtest_discover_tests(openturbine_unit_tests PROPERTIES DISCOVERY_TIMEOUT 60000)
