# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_beams_input.cpp
    test_beam_quadrature.cpp
    test_generate_sectional_properties.cpp
    test_hollow_circle_properties.cpp
    test_interpolate_QP_state.cpp
    test_interpolate_QP_vector.cpp
    test_interpolation.cpp
)
