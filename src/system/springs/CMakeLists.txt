target_sources(openturbine_library PRIVATE)

install(FILES
        calculate_distance_components.hpp
        calculate_force_coefficients.hpp
        calculate_force_vectors.hpp
        calculate_length.hpp
        calculate_stiffness_matrix.hpp
        update_node_state.hpp
        DESTINATION include/OpenTurbine/system/springs/
)
