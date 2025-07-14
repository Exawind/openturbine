#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

namespace openturbine::beams {

template <typename DeviceType>
struct IntegrateInertiaMatrixElement {
    size_t element;
    size_t num_nodes;
    size_t num_qps;
    typename Kokkos::View<double*, DeviceType>::const_type qp_weight_;
    typename Kokkos::View<double*, DeviceType>::const_type qp_jacobian_;
    typename Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>::const_type shape_interp_;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type qp_Muu_;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type qp_Guu_;
    double beta_prime_;
    double gamma_prime_;
    Kokkos::View<double** [6][6], DeviceType> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(size_t node_simd_node) const {
        using simd_type = Kokkos::Experimental::simd<double>;
        using tag_type = Kokkos::Experimental::vector_aligned_tag;
        constexpr auto width = simd_type::size();
        const auto extra_component = num_nodes % width == 0U ? 0U : 1U;
        const auto num_simd_nodes = num_nodes / width + extra_component;
        const auto node = node_simd_node / num_simd_nodes;
        const auto simd_node = (node_simd_node % num_simd_nodes) * width;

        auto local_M = Kokkos::Array<simd_type, 36>{};

        const auto qp_Muu =
            typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Muu_.data(), num_qps);
        const auto qp_Guu =
            typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Guu_.data(), num_qps);

        for (auto qp = 0U; qp < num_qps; ++qp) {
            const auto w = simd_type(qp_weight_(qp));
            const auto jacobian = simd_type(qp_jacobian_(qp));
            const auto phi_1 = simd_type(shape_interp_(node, qp));
            auto phi_2 = simd_type{};
            phi_2.copy_from(&shape_interp_(simd_node, qp), tag_type());
            const auto coeff = phi_1 * phi_2 * w * jacobian;
            const auto Muu_local = Kokkos::subview(qp_Muu, qp, Kokkos::ALL);
            const auto Guu_local = Kokkos::subview(qp_Guu, qp, Kokkos::ALL);
            for (auto component = 0; component < 36; ++component) {
                const auto contribution = simd_type(
                    beta_prime_ * Muu_local(component) + gamma_prime_ * Guu_local(component)
                );
                local_M[component] = local_M[component] + coeff * contribution;
            }
        }

        const auto num_lanes = Kokkos::min(width, num_nodes - simd_node);
        const auto global_M =
            Kokkos::View<double** [36], DeviceType>(gbl_M_.data(), num_nodes, num_nodes);
        const auto M_slice = Kokkos::subview(
            global_M, node, Kokkos::make_pair(simd_node, simd_node + num_lanes), Kokkos::ALL
        );

        for (auto lane = 0U; lane < num_lanes; ++lane) {
            for (auto component = 0; component < 36; ++component) {
                M_slice(lane, component) = local_M[component][lane];
            }
        }
    }
};
}  // namespace openturbine::beams
