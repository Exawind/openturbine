# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_interpolate_QP_state.cpp
    test_interpolate_QP_vector.cpp
)
