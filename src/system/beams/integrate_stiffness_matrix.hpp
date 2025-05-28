#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

namespace openturbine::beams {

template <typename DeviceType>
struct IntegrateStiffnessMatrixElement {
    size_t i_elem;
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
    void operator()(size_t ij_index) const {
        using simd_type = Kokkos::Experimental::simd<double>;
        using mask_type = Kokkos::Experimental::simd_mask<double>;
        using tag_type = Kokkos::Experimental::element_aligned_tag;
        constexpr auto width = simd_type::size();
        const auto extra_component = num_nodes % width == 0U ? 0U : 1U;
        const auto simd_nodes = num_nodes / width + extra_component;
        const auto i_index = ij_index / simd_nodes;
        const auto j_index = (ij_index % simd_nodes) * width;

        auto mask = mask_type([j_index, final_nodes = this->num_nodes](size_t lane) {
            return j_index + lane < final_nodes;
        });
        auto local_M = Kokkos::Array<simd_type, 36>{};

        const auto qp_Kuu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Kuu_.data(), num_qps);
        const auto qp_Puu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Puu_.data(), num_qps);
        const auto qp_Cuu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Cuu_.data(), num_qps);
        const auto qp_Ouu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Ouu_.data(), num_qps);
        const auto qp_Quu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Quu_.data(), num_qps);
        const auto gbl_M = Kokkos::View<double** [36], DeviceType>(gbl_M_.data(), num_nodes, num_nodes);
        
        for (auto k = 0U; k < num_qps; ++k) {
            const auto w = simd_type(qp_weight_(k));
            const auto jacobian = simd_type(qp_jacobian_(k));
            const auto phi_i = simd_type(shape_interp_(i_index, k));
            auto phi_j = simd_type{};
            Kokkos::Experimental::where(mask, phi_j)
                .copy_from(&shape_interp_(j_index, k), tag_type());
            const auto phi_prime_i = simd_type(shape_deriv_(i_index, k));
            auto phi_prime_j = simd_type{};
            Kokkos::Experimental::where(mask, phi_prime_j)
                .copy_from(&shape_deriv_(j_index, k), tag_type());
            const auto K = (phi_i * phi_j) * (w * jacobian);
            const auto P = (phi_i * phi_prime_j) * w;
            const auto C = (phi_prime_i * phi_prime_j) * (w / jacobian);
            const auto O = (phi_prime_i * phi_j) * w;
            for (auto m = 0; m < 36; ++m) {
                local_M[m] = local_M[m] + K * simd_type(qp_Kuu(k, m) + qp_Quu(k, m)) + P * simd_type(qp_Puu(k, m)) + C * simd_type(qp_Cuu(k, m)) + O * simd_type(qp_Ouu(k, m));
            }
        }
        for (auto lane = 0U; lane < width && mask[lane]; ++lane) {
            for (auto m = 0; m < 36; ++m) {
                gbl_M(i_index, j_index + lane, m) = local_M[m][lane];
            }
        }
    }
};

}  // namespace openturbine::beams
