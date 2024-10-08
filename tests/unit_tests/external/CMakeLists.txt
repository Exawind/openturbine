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
endif()