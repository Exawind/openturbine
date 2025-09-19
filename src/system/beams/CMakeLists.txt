target_sources(kynema_library PRIVATE)

install(FILES
        calculate_Ouu.hpp
        calculate_Puu.hpp
        calculate_Quu.hpp
        calculate_force_FC.hpp
        calculate_force_FD.hpp
        calculate_inertial_quadrature_point_values.hpp
        calculate_quadrature_point_values.hpp
        calculate_stiffness_quadrature_point_values.hpp
        calculate_strain.hpp
        calculate_system_matrix.hpp
        calculate_temporary_variables.hpp
        integrate_inertia_matrix.hpp
        integrate_residual_vector.hpp
        integrate_stiffness_matrix.hpp
        interpolate_to_quadrature_point_for_inertia.hpp
        interpolate_to_quadrature_point_for_stiffness.hpp
        update_node_state.hpp
        DESTINATION include/Kynema/system/beams/
)
