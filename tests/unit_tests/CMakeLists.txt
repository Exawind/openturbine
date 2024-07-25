
add_executable(openturbine_unit_tests)

add_subdirectory(restruct_poc)

target_sources(
  openturbine_unit_tests
  PRIVATE
  utest_main.cpp
)

target_link_libraries(
  openturbine_unit_tests 
  PRIVATE 
  openturbine_warnings 
  openturbine_options 
  openturbine_library)

target_link_system_libraries(
  openturbine_unit_tests
  PRIVATE 
  KokkosKernels::kokkoskernels
  Amesos2::amesos2
  GTest::gtest 
  GTest::gtest_main)

if (OTURB_ENABLE_VTK)
  target_link_system_libraries(
    openturbine_unit_tests 
    PRIVATE
    VTK::IOXML
  )
endif()

include(GoogleTest)
gtest_discover_tests(openturbine_unit_tests PROPERTIES DISCOVERY_TIMEOUT 60000)