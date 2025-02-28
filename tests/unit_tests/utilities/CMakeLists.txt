add_subdirectory(netcdf)

target_sources(
    openturbine_unit_tests
    PRIVATE
    test_turbine_parsing.cpp
)
