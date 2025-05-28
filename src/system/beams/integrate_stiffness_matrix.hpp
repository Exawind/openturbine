#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

namespace openturbine::beams {

template <typename DeviceType>
struct IntegrateStiffnessMatrixElement {
    size_t element;
    size_t num_nodes;
    size_t num_qps;
    typename Kokkos::View<double*, DeviceType>::const_type qp_weight_;
    typename Kokkos::View<double*, DeviceType>::const_type qp_jacobian_;
    typename Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>::const_type shape_interp_;
    typename Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>::const_type shape_deriv_;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type qp_Kuu_;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type qp_Puu_;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type qp_Cuu_;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type qp_Ouu_;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type qp_Quu_;
    Kokkos::View<double** [6][6], DeviceType> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(size_t node_simd_node) const {
        using simd_type = Kokkos::Experimental::simd<double>;
        using tag_type = Kokkos::Experimental::vector_aligned_tag;
        constexpr auto width = simd_type::size();
        const auto extra_component = num_nodes % width == 0U ? 0U : 1U;
        const auto simd_nodes = num_nodes / width + extra_component;
        const auto node = node_simd_node / simd_nodes;
        const auto simd_node = (node_simd_node % simd_nodes) * width;

        auto local_M = Kokkos::Array<simd_type, 36>{};

        const auto qp_Kuu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Kuu_.data(), num_qps);
        const auto qp_Puu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Puu_.data(), num_qps);
        const auto qp_Cuu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Cuu_.data(), num_qps);
        const auto qp_Ouu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Ouu_.data(), num_qps);
        const auto qp_Quu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Quu_.data(), num_qps);
        const auto gbl_M = Kokkos::View<double** [36], DeviceType>(gbl_M_.data(), num_nodes, num_nodes);
        
        for (auto qp = 0U; qp < num_qps; ++qp) {
            const auto w = simd_type(qp_weight_(qp));
            const auto jacobian = simd_type(qp_jacobian_(qp));
            const auto phi_i = simd_type(shape_interp_(node, qp));
            auto phi_j = simd_type{};
            phi_j.copy_from(&shape_interp_(simd_node, qp), tag_type());
            const auto phi_prime_i = simd_type(shape_deriv_(node, qp));
            auto phi_prime_j = simd_type{};
            phi_prime_j.copy_from(&shape_deriv_(simd_node, qp), tag_type());
            const auto K = (phi_i * phi_j) * (w * jacobian);
            const auto P = (phi_i * phi_prime_j) * w;
            const auto C = (phi_prime_i * phi_prime_j) * (w / jacobian);
            const auto O = (phi_prime_i * phi_j) * w;
            for (auto component = 0; component < 36; ++component) {
                local_M[component] = local_M[component] + K * simd_type(qp_Kuu(qp, component) + qp_Quu(qp, component)) + P * simd_type(qp_Puu(qp, component)) + C * simd_type(qp_Cuu(qp, component)) + O * simd_type(qp_Ouu(qp, component));
            }
        }
        for (auto lane = 0U; lane < width && simd_node + lane < num_nodes; ++lane) {
            for (auto component = 0; component < 36; ++component) {
                gbl_M(node, simd_node + lane, component) = local_M[component][lane];
            }
        }
    }
};

}  // namespace openturbine::beams
