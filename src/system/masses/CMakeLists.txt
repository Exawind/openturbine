target_sources(openturbine_library PRIVATE)

install(FILES
        calculate_gravity_force.hpp
        calculate_gyroscopic_matrix.hpp
        calculate_inertial_force.hpp
        calculate_inertia_stiffness_matrix.hpp
        calculate_mass_matrix_components.hpp
        calculate_QP_position.hpp
        calculate_RR0.hpp
        rotate_section_matrix.hpp
        update_node_state.hpp
        DESTINATION include/OpenTurbine/system/masses
)
