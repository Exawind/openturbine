add_subdirectory(beams)
add_subdirectory(masses)
add_subdirectory(springs)

# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_elements.cpp
)
