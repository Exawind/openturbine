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
        auto local_M_data = Kokkos::Array<simd_type, 36>{};
        const auto local_M = Kokkos::View<simd_type[6][6], DeviceType>(local_M_data.data());
        for (auto k = 0U; k < num_qps; ++k) {
            const auto w = simd_type(qp_weight_(k));
            const auto jacobian = simd_type(qp_jacobian_(k));
            const auto phi_i = simd_type(shape_interp_(i_index, k));
            auto phi_j = simd_type{};
            Kokkos::Experimental::where(mask, phi_j)
                .copy_from(&shape_interp_(j_index, k), tag_type());
            const auto coeff = phi_i * phi_j * w * jacobian;
            for (auto m = 0U; m < 6U; ++m) {
                local_M(m, 0) =
                    local_M(m, 0) +
                    coeff *
                        simd_type(beta_prime_ * qp_Muu_(k, m, 0) + gamma_prime_ * qp_Guu_(k, m, 0));
                local_M(m, 1) =
                    local_M(m, 1) +
                    coeff *
                        simd_type(beta_prime_ * qp_Muu_(k, m, 1) + gamma_prime_ * qp_Guu_(k, m, 1));
                local_M(m, 2) =
                    local_M(m, 2) +
                    coeff *
                        simd_type(beta_prime_ * qp_Muu_(k, m, 2) + gamma_prime_ * qp_Guu_(k, m, 2));
                local_M(m, 3) =
                    local_M(m, 3) +
                    coeff *
                        simd_type(beta_prime_ * qp_Muu_(k, m, 3) + gamma_prime_ * qp_Guu_(k, m, 3));
                local_M(m, 4) =
                    local_M(m, 4) +
                    coeff *
                        simd_type(beta_prime_ * qp_Muu_(k, m, 4) + gamma_prime_ * qp_Guu_(k, m, 4));
                local_M(m, 5) =
                    local_M(m, 5) +
                    coeff *
                        simd_type(beta_prime_ * qp_Muu_(k, m, 5) + gamma_prime_ * qp_Guu_(k, m, 5));
            }
        }
        for (auto lane = 0U; lane < width && mask[lane]; ++lane) {
            for (auto m = 0U; m < 6U; ++m) {
                gbl_M_(i_index, j_index + lane, m, 0) = local_M(m, 0)[lane];
                gbl_M_(i_index, j_index + lane, m, 1) = local_M(m, 1)[lane];
                gbl_M_(i_index, j_index + lane, m, 2) = local_M(m, 2)[lane];
                gbl_M_(i_index, j_index + lane, m, 3) = local_M(m, 3)[lane];
                gbl_M_(i_index, j_index + lane, m, 4) = local_M(m, 4)[lane];
                gbl_M_(i_index, j_index + lane, m, 5) = local_M(m, 5)[lane];
            }
        }
    }
};
}  // namespace openturbine::beams
