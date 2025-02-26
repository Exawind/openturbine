# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    test_netcdf_utilities.cpp
    test_turbine_parsing.cpp
)
