#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

namespace openturbine::beams {

template <typename DeviceType>
struct IntegrateInertiaMatrixElement {
    size_t i_elem;
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
    void operator()(size_t ij_index) const {
        using simd_type = Kokkos::Experimental::simd<double>;
        using mask_type = Kokkos::Experimental::simd_mask<double>;
        using tag_type = Kokkos::Experimental::element_aligned_tag;
        constexpr auto width = simd_type::size();
        const auto extra_component = num_nodes % width == 0U ? 0U : 1U;
        const auto simd_nodes = num_nodes / width + extra_component;
        const auto i_index = ij_index / simd_nodes;
        const auto j_index = (ij_index % simd_nodes) * width;

        auto mask = mask_type([j_index, final_node = this->num_nodes](size_t lane) {
            return j_index + lane < final_node;
        });
        auto local_M = Kokkos::Array<simd_type, 36>{};
        
        const auto qp_Muu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Muu_.data(), num_qps);
        const auto qp_Guu = typename Kokkos::View<double* [36], DeviceType>::const_type(qp_Guu_.data(), num_qps);
        const auto gbl_M = Kokkos::View<double** [36], DeviceType>(gbl_M_.data(), num_nodes, num_nodes);

        for (auto k = 0U; k < num_qps; ++k) {
            const auto w = simd_type(qp_weight_(k));
            const auto jacobian = simd_type(qp_jacobian_(k));
            const auto phi_i = simd_type(shape_interp_(i_index, k));
            auto phi_j = simd_type{};
            Kokkos::Experimental::where(mask, phi_j)
                .copy_from(&shape_interp_(j_index, k), tag_type());
            const auto coeff = phi_i * phi_j * w * jacobian;
            for (auto m = 0; m < 36; ++m) {
                local_M[m] = local_M[m] + coeff * simd_type(beta_prime_ * qp_Muu(k, m) + gamma_prime_ * qp_Guu(k, m));
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
