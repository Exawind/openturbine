# Specify the ROSCO controller test source files for the unit test executable
if (OpenTurbine_BUILD_ROSCO_CONTROLLER)
    target_sources(
        openturbine_unit_tests
        PRIVATE
        test_rosco_controller.cpp
    )
endif()


