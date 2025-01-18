target_sources(openturbine_library PRIVATE)

install(FILES
        calculate_force_FC.hpp
        calculate_force_FD.hpp
        calculate_gravity_force.hpp
        calculate_gyroscopic_matrix.hpp
        calculate_inertial_force.hpp
        calculate_inertia_stiffness_matrix.hpp
        calculate_mass_matrix_components.hpp
        calculate_Ouu.hpp
        calculate_Puu.hpp
        calculate_quadrature_point_values.hpp
        calculate_Quu.hpp
        calculate_RR0.hpp
        calculate_strain.hpp
        calculate_temporary_variables.hpp
        integrate_inertia_matrix.hpp
        integrate_residual_vector.hpp
        integrate_stiffness_matrix.hpp
        rotate_section_matrix.hpp
        update_node_state.hpp
        DESTINATION include/OpenTurbine/system/beams/
)
