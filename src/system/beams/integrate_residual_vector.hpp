#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::beams {

template <typename DeviceType>
struct IntegrateResidualVectorElement {
    size_t element;
    size_t num_qps;
    typename Kokkos::View<double*, DeviceType>::const_type qp_weight_;
    typename Kokkos::View<double*, DeviceType>::const_type qp_jacobian_;
    typename Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>::const_type shape_interp_;
    typename Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>::const_type shape_deriv_;
    typename Kokkos::View<double* [6], DeviceType>::const_type node_FX_;
    typename Kokkos::View<double* [6], DeviceType>::const_type qp_Fc_;
    typename Kokkos::View<double* [6], DeviceType>::const_type qp_Fd_;
    typename Kokkos::View<double* [6], DeviceType>::const_type qp_Fi_;
    typename Kokkos::View<double* [6], DeviceType>::const_type qp_Fe_;
    typename Kokkos::View<double* [6], DeviceType>::const_type qp_Fg_;
    Kokkos::View<double** [6], DeviceType> residual_vector_terms_;

    KOKKOS_FUNCTION
    void operator()(size_t node) const {
        auto local_residual = Kokkos::Array<double, 6>{};
        for (auto qp = 0U; qp < num_qps; ++qp) {
            const auto weight = qp_weight_(qp);
            const auto coeff_c = weight * shape_deriv_(node, qp);
            const auto coeff_dig = weight * qp_jacobian_(qp) * shape_interp_(node, qp);
            for (auto component = 0U; component < 6U; ++component) {
                local_residual[component] += coeff_c * qp_Fc_(qp, component) +
                                         coeff_dig * (qp_Fd_(qp, component) + qp_Fi_(qp, component) -
                                                      qp_Fe_(qp, component) - qp_Fg_(qp, component));
            }
        }
        for (auto component = 0U; component < 6U; ++component) {
            residual_vector_terms_(element, node, component) =
                local_residual[component] - node_FX_(node, component);
        }
    }
};

}  // namespace openturbine::beams
