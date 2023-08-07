
# Add subdirectories for unit tests
add_subdirectory(rigid_pendulum_poc)

target_sources(
    ${oturb_unit_test_exe_name}
    PRIVATE
    utest_main.cpp
    test_config.cpp
    test_log.cpp
)

target_compile_options(
    ${oturb_unit_test_exe_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${OTURB_CXX_FLAGS}>
)
target_include_directories(${oturb_unit_test_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${oturb_unit_test_exe_name} PRIVATE ${PROJECT_BINARY_DIR})

# Link our unit test executable with GoogleTest
find_package(GTest REQUIRED)
target_link_libraries(${oturb_unit_test_exe_name} PRIVATE GTest::gtest GTest::gtest_main)

# Link to OpenTurbine test targets
target_link_libraries(${oturb_unit_test_exe_name} PRIVATE ${oturb_lib_name})

# Link Kokkos to test target
find_package(Kokkos REQUIRED)
find_package(LAPACK REQUIRED)
target_link_libraries(${oturb_unit_test_exe_name} PRIVATE Kokkos::kokkos LAPACK::LAPACK lapacke lapack blas)

# Define what we want to be installed during a make install
install(TARGETS ${oturb_unit_test_exe_name}
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
