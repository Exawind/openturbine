# Test ROSCO controller
if (OpenTurbine_BUILD_ROSCO_CONTROLLER AND
    (NOT OpenTurbine_ENABLE_SANITIZER_LEAK))
    target_sources(
        openturbine_unit_tests
        PRIVATE
        test_rosco_controller.cpp
    )
endif()

# Test AeroDyn Inflow
# if (OpenTurbine_BUILD_OPENFAST_ADI AND
#     (NOT OpenTurbine_ENABLE_SANITIZER_LEAK))
#     target_sources(
#         openturbine_unit_tests
#         PRIVATE
#         test_aerodyn_inflow.cpp
#     )
# endif()

# Test AeroDyn Inflow
if (OpenTurbine_BUILD_OPENFAST_ADI AND
    OpenTurbine_BUILD_ROSCO_CONTROLLER AND
    (NOT OpenTurbine_ENABLE_SANITIZER_LEAK))
    target_sources(
        openturbine_unit_tests
        PRIVATE
        test_rotor_aero_controller.cpp
    )
    add_custom_target(
        copy_external_test_files ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/IEA-15-240-RWT
                $<TARGET_FILE_DIR:openturbine_unit_tests>/IEA-15-240-RWT)
endif()