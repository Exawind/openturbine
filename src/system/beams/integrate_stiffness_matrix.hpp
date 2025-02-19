#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

namespace openturbine::beams {
struct IntegrateStiffnessMatrixElement {
    size_t i_elem;
    size_t num_nodes;
    size_t num_qps;
    Kokkos::View<double*>::const_type qp_weight_;
    Kokkos::View<double*>::const_type qp_jacobian_;
    Kokkos::View<double**, Kokkos::LayoutLeft>::const_type shape_interp_;
    Kokkos::View<double**, Kokkos::LayoutLeft>::const_type shape_deriv_;
    Kokkos::View<double* [6][6]>::const_type qp_Kuu_;
    Kokkos::View<double* [6][6]>::const_type qp_Puu_;
    Kokkos::View<double* [6][6]>::const_type qp_Cuu_;
    Kokkos::View<double* [6][6]>::const_type qp_Ouu_;
    Kokkos::View<double* [6][6]>::const_type qp_Quu_;
    Kokkos::View<double*** [6][6]> gbl_M_;

    KOKKOS_FUNCTION
    void operator()(size_t ij_index) const {
        using simd_type = Kokkos::Experimental::native_simd<double>;
        using mask_type = Kokkos::Experimental::native_simd_mask<double>;
        using tag_type = Kokkos::Experimental::element_aligned_tag;
        constexpr auto width = simd_type::size();
        const auto extra_component = num_nodes % width == 0U ? 0U : 1U;
        const auto simd_nodes = num_nodes / width + extra_component;
        const auto i_index = ij_index / simd_nodes;
        const auto j_index = (ij_index % simd_nodes) * width;

        auto mask = mask_type([j_index, final_nodes = this->num_nodes](size_t lane) {
            return j_index + lane < final_nodes;
        });
        auto local_M_data = Kokkos::Array<simd_type, 36>{};
        const auto local_M = Kokkos::View<simd_type[6][6]>(local_M_data.data());
        for (auto k = 0U; k < num_qps; ++k) {
            const auto w = qp_weight_(k);
            const auto jacobian = qp_jacobian_(k);
            const auto phi_i = shape_interp_(i_index, k);
            auto phi_j = simd_type{};
            Kokkos::Experimental::where(mask, phi_j)
                .copy_from(&shape_interp_(j_index, k), tag_type());
            const auto phi_prime_i = shape_deriv_(i_index, k);
            auto phi_prime_j = simd_type{};
            Kokkos::Experimental::where(mask, phi_prime_j)
                .copy_from(&shape_deriv_(j_index, k), tag_type());
            const auto K = w * phi_i * phi_j * jacobian;
            const auto P = w * (phi_i * phi_prime_j);
            const auto C = w * (phi_prime_i * phi_prime_j / jacobian);
            const auto O = w * (phi_prime_i * phi_j);
            for (auto m = 0U; m < 6U; ++m) {
                local_M(m, 0) += K * (qp_Kuu_(k, m, 0) + qp_Quu_(k, m, 0)) + P * qp_Puu_(k, m, 0) +
                                 C * qp_Cuu_(k, m, 0) + O * qp_Ouu_(k, m, 0);
                local_M(m, 1) += K * (qp_Kuu_(k, m, 1) + qp_Quu_(k, m, 1)) + P * qp_Puu_(k, m, 1) +
                                 C * qp_Cuu_(k, m, 1) + O * qp_Ouu_(k, m, 1);
                local_M(m, 2) += K * (qp_Kuu_(k, m, 2) + qp_Quu_(k, m, 2)) + P * qp_Puu_(k, m, 2) +
                                 C * qp_Cuu_(k, m, 2) + O * qp_Ouu_(k, m, 2);
                local_M(m, 3) += K * (qp_Kuu_(k, m, 3) + qp_Quu_(k, m, 3)) + P * qp_Puu_(k, m, 3) +
                                 C * qp_Cuu_(k, m, 3) + O * qp_Ouu_(k, m, 3);
                local_M(m, 4) += K * (qp_Kuu_(k, m, 4) + qp_Quu_(k, m, 4)) + P * qp_Puu_(k, m, 4) +
                                 C * qp_Cuu_(k, m, 4) + O * qp_Ouu_(k, m, 4);
                local_M(m, 5) += K * (qp_Kuu_(k, m, 5) + qp_Quu_(k, m, 5)) + P * qp_Puu_(k, m, 5) +
                                 C * qp_Cuu_(k, m, 5) + O * qp_Ouu_(k, m, 5);
            }
        }
        for (auto lane = 0U; lane < width && mask[lane]; ++lane) {
            for (auto m = 0U; m < 6U; ++m) {
                gbl_M_(i_elem, i_index, j_index + lane, m, 0) = local_M(m, 0)[lane];
                gbl_M_(i_elem, i_index, j_index + lane, m, 1) = local_M(m, 1)[lane];
                gbl_M_(i_elem, i_index, j_index + lane, m, 2) = local_M(m, 2)[lane];
                gbl_M_(i_elem, i_index, j_index + lane, m, 3) = local_M(m, 3)[lane];
                gbl_M_(i_elem, i_index, j_index + lane, m, 4) = local_M(m, 4)[lane];
                gbl_M_(i_elem, i_index, j_index + lane, m, 5) = local_M(m, 5)[lane];
            }
        }
    }
};
}  // namespace openturbine::beams
