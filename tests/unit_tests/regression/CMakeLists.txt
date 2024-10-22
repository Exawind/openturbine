# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_beams.cpp
    test_solver.cpp
    test_rotating_beam.cpp
    test_cantilever_beam.cpp
    test_rotor.cpp
    test_utilities.cpp
    test_controller.cpp
)
