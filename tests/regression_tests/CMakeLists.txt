# Add the executable target for regression tests
add_executable(openturbine_regression_tests)

# Add subdirectories for additional components
add_subdirectory(external)
add_subdirectory(regression)

# Specify the source files for the regression tests
target_sources(openturbine_regression_tests PRIVATE rtest_main.cpp)

# Link internal libraries to the regression test executable
target_link_libraries(openturbine_regression_tests PRIVATE
  openturbine_warnings
  openturbine_options
  OpenTurbine::openturbine_library
)

# Link system libraries and dependencies
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_TARGET yaml-cpp)
endif()

target_link_system_libraries(openturbine_regression_tests PRIVATE
  KokkosKernels::kokkoskernels
  Amesos2::amesos2
  ${YAML_CPP_TARGET}
  GTest::gtest
  GTest::gtest_main
)

# Conditional linking for VTK if enabled
if(OpenTurbine_ENABLE_VTK)
  target_link_system_libraries(openturbine_regression_tests PRIVATE VTK::IOXML)
  target_compile_definitions(openturbine_regression_tests PRIVATE OpenTurbine_ENABLE_VTK)
endif()

# Add compile definitions for OpenTurbine_BUILD_OPENFAST_ADI if enabled
if(OpenTurbine_BUILD_OPENFAST_ADI)
  target_compile_definitions(openturbine_regression_tests PRIVATE OpenTurbine_BUILD_OPENFAST_ADI)
endif()

# Include GoogleTest module and discover tests
include(GoogleTest)
gtest_discover_tests(openturbine_regression_tests PROPERTIES DISCOVERY_TIMEOUT 60000)
