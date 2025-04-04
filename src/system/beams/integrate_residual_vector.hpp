#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::beams {

struct IntegrateResidualVectorElement {
    size_t i_elem;
    size_t num_qps;
    Kokkos::View<double*>::const_type qp_weight_;
    Kokkos::View<double*>::const_type qp_jacobian_;
    Kokkos::View<double**, Kokkos::LayoutLeft>::const_type shape_interp_;
    Kokkos::View<double**, Kokkos::LayoutLeft>::const_type shape_deriv_;
    Kokkos::View<double* [6]>::const_type node_FX_;
    Kokkos::View<double* [6]>::const_type qp_Fc_;
    Kokkos::View<double* [6]>::const_type qp_Fd_;
    Kokkos::View<double* [6]>::const_type qp_Fi_;
    Kokkos::View<double* [6]>::const_type qp_Fe_;
    Kokkos::View<double* [6]>::const_type qp_Fg_;
    Kokkos::View<double** [6]> residual_vector_terms_;

    KOKKOS_FUNCTION
    void operator()(size_t i_node) const {
        auto local_residual = Kokkos::Array<double, 6>{};
        for (auto j_qp = 0U; j_qp < num_qps; ++j_qp) {
            const auto weight = qp_weight_(j_qp);
            const auto coeff_c = weight * shape_deriv_(i_node, j_qp);
            const auto coeff_dig = weight * qp_jacobian_(j_qp) * shape_interp_(i_node, j_qp);
            for (auto k_dof = 0U; k_dof < 6U; ++k_dof) {
                local_residual[k_dof] += coeff_c * qp_Fc_(j_qp, k_dof) +
                                         coeff_dig * (qp_Fd_(j_qp, k_dof) + qp_Fi_(j_qp, k_dof) -
                                                      qp_Fe_(j_qp, k_dof) - qp_Fg_(j_qp, k_dof));
            }
        }
        for (auto k_dof = 0U; k_dof < 6U; ++k_dof) {
            residual_vector_terms_(i_elem, i_node, k_dof) =
                local_residual[k_dof] - node_FX_(i_node, k_dof);
        }
    }
};

}  // namespace openturbine::beams
