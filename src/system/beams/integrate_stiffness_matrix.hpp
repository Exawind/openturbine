#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBatched_Copy_Decl.hpp>
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

        const auto qp_Kuu =
            typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Kuu_.data(), num_qps);
        const auto qp_Puu =
            typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Puu_.data(), num_qps);
        const auto qp_Cuu =
            typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Cuu_.data(), num_qps);
        const auto qp_Ouu =
            typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Ouu_.data(), num_qps);
        const auto qp_Quu =
            typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Quu_.data(), num_qps);

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
	    const auto Kuu_local = Kokkos::subview(qp_Kuu, qp, Kokkos::ALL);
	    const auto Quu_local = Kokkos::subview(qp_Quu, qp, Kokkos::ALL);
	    const auto Puu_local = Kokkos::subview(qp_Puu, qp, Kokkos::ALL);
	    const auto Cuu_local = Kokkos::subview(qp_Cuu, qp, Kokkos::ALL);
	    const auto Ouu_local = Kokkos::subview(qp_Ouu, qp, Kokkos::ALL);
            for (auto component = 0; component < 36; ++component) {
		const auto Kuu = simd_type(Kuu_local(component));
		const auto Quu = simd_type(Quu_local(component));
		const auto Puu = simd_type(Puu_local(component));
		const auto Cuu = simd_type(Cuu_local(component));
		const auto Ouu = simd_type(Ouu_local(component));
		local_M[component] = local_M[component] + K * (Kuu + Quu) + P * Puu + C * Cuu + O * Ouu;
            }
        }
 
	const auto num_lanes = Kokkos::min(width, num_nodes-simd_node);
        const auto global_M =
            Kokkos::View<double** [36], DeviceType>(gbl_M_.data(), num_nodes, num_nodes);
	const auto M_slice = Kokkos::subview(global_M, node, Kokkos::make_pair(simd_node, simd_node + num_lanes), Kokkos::ALL);

        for (auto lane = 0U; lane < num_lanes; ++lane) {
            for (auto component = 0; component < 36; ++component) {
                M_slice(lane, component) = local_M[component][lane];
            }
        }
	
    }
};

}  // namespace openturbine::beams
