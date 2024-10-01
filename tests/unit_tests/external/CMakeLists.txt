# Test ROSCO controller
if (OpenTurbine_BUILD_ROSCO_CONTROLLER AND 
    (NOT OpenTurbine_ENABLE_SANITIZER_LEAK))
    target_sources(
        openturbine_unit_tests
        PRIVATE
        test_rosco_controller.cpp
    )
endif()


# Full IEA15 Rotor with Aero and Controller
if (OpenTurbine_BUILD_ROSCO_CONTROLLER AND OpenTurbine_BUILD_OPENFAST_ADI AND 
    (NOT OpenTurbine_ENABLE_SANITIZER_LEAK))
    target_sources(
        openturbine_unit_tests
        PRIVATE
        test_rotor_aero_controller.cpp
    )
    add_custom_target(unit_test_input_files ALL)
    add_custom_command(
        TARGET unit_test_input_files
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/Cp_Ct_Cq.IEA15MW.txt
                ${CMAKE_CURRENT_SOURCE_DIR}/IEA-15-240-RWT_AeroDyn15_blade.dat
                ${CMAKE_CURRENT_SOURCE_DIR}/IEA-15-240-RWT_InflowFile.dat
                ${CMAKE_CURRENT_SOURCE_DIR}/IEA-15-240-RWT_AeroDyn15.dat
                ${CMAKE_CURRENT_SOURCE_DIR}/IEA-15-240-RWT_DISCON.IN
                ${CMAKE_BINARY_DIR}/tests/unit_tests
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/Airfoils
                ${CMAKE_BINARY_DIR}/tests/unit_tests/Airfoils)
endif()


