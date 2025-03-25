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
    void operator()(size_t i_index) const {
        auto local_residual = Kokkos::Array<double, 6>{};
        for (auto j_index = 0U; j_index < num_qps; ++j_index) {  // QPs
            const auto weight = qp_weight_(j_index);
            const auto coeff_c = weight * shape_deriv_(i_index, j_index);
            const auto coeff_dig = weight * qp_jacobian_(j_index) * shape_interp_(i_index, j_index);
            for (auto k = 0U; k < 6U; ++k) {
                local_residual[k] += coeff_c * qp_Fc_(j_index, k) +
                                     coeff_dig * (qp_Fd_(j_index, k) + qp_Fi_(j_index, k) -
                                                  qp_Fe_(j_index, k) - qp_Fg_(j_index, k));
            }
        }
        for (auto k = 0U; k < 6U; ++k) {
            residual_vector_terms_(i_elem, i_index, k) = local_residual[k] - node_FX_(i_index, k);
        }
    }
};

}  // namespace openturbine::beams
