# Specify the source files for the unit test executable
target_sources(
    openturbine_unit_tests
    PRIVATE
    # Model-related tests
    test_model.cpp
    test_copy_nodes_to_state.cpp
    test_mesh_connectivity.cpp
)
