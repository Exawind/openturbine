# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
)

add_subdirectory(springs)
add_subdirectory(beams)
add_subdirectory(masses)
