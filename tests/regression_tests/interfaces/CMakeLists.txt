# Specify the source files for the regression test executable
target_sources(
    openturbine_regression_tests
    PRIVATE
    test_cfd_interface.cpp
    test_blade_interface.cpp
    test_turbine_interface.cpp
)

add_custom_target(
    copy_interfaces_test_files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/interfaces_test_files
            $<TARGET_FILE_DIR:openturbine_regression_tests>/interfaces_test_files)
