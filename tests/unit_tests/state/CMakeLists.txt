# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_calculate_next_state.cpp
    test_calculate_displacement.cpp
    test_update_algorithmic_acceleration.cpp
    test_update_dynamic_prediction.cpp
    test_update_global_position.cpp
)
