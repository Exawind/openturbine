#pragma once

#include <Kokkos_Core.hpp>

#include "system/springs/calculate_distance_components.hpp"
#include "system/springs/calculate_force_coefficients.hpp"
#include "system/springs/calculate_force_vectors.hpp"
#include "system/springs/calculate_length.hpp"
#include "system/springs/calculate_stiffness_matrix.hpp"

namespace openturbine::springs {

struct CalculateQuadraturePointValues {
    Kokkos::View<double* [7]>::const_type Q;

    Kokkos::View<size_t* [2]>::const_type node_state_indices;
    Kokkos::View<double* [3]>::const_type x0_;
    Kokkos::View<double*>::const_type l_ref_;
    Kokkos::View<double*>::const_type k_;

    Kokkos::View<double* [2][3]> residual_vector_terms;
    Kokkos::View<double* [2][2][3][3]> stiffness_matrix_terms;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        const auto index_0 = node_state_indices(i_elem, 0);
        const auto index_1 = node_state_indices(i_elem, 1);

        const auto x0_data =
            Kokkos::Array<double, 3>{x0_(i_elem, 0), x0_(i_elem, 1), x0_(i_elem, 2)};
        const auto u1_data = Kokkos::Array<double, 3>{Q(index_0, 0), Q(index_0, 1), Q(index_0, 2)};
        const auto u2_data = Kokkos::Array<double, 3>{Q(index_1, 0), Q(index_1, 1), Q(index_1, 2)};
        auto r_data = Kokkos::Array<double, 3>{};
        auto f_data = Kokkos::Array<double, 3>{};
        auto a_data = Kokkos::Array<double, 9>{};

        const auto x0 = Kokkos::View<double[3]>::const_type(x0_data.data());
        const auto u1 = Kokkos::View<double[3]>::const_type(u1_data.data());
        const auto u2 = Kokkos::View<double[3]>::const_type(u2_data.data());
        const auto r = Kokkos::View<double[3]>(r_data.data());
        const auto f = Kokkos::View<double[3]>(f_data.data());
        const auto a = Kokkos::View<double[3][3]>(a_data.data());

        const auto l_ref = l_ref_(i_elem);
        const auto k = k_(i_elem);

        springs::CalculateDistanceComponents(x0, u1, u2, r);
        const auto l = springs::CalculateLength(r);
        const auto c1 = springs::CalculateForceCoefficient1(k, l_ref, l);
        const auto c2 = springs::CalculateForceCoefficient2(k, l_ref, l);
        CalculateForceVectors(r, c1, f);
        CalculateStiffnessMatrix(c1, c2, r, l, a);

        for (auto i = 0U; i < 3U; ++i) {
            residual_vector_terms(i_elem, 0, i) = f(i);
            residual_vector_terms(i_elem, 1, i) = -f(i);
        }

        for (auto i = 0U; i < 3U; ++i) {
            for (auto j = 0U; j < 3U; ++j) {
                stiffness_matrix_terms(i_elem, 0, 0, i, j) = a(i, j);
            }
        }
        for (auto i = 0U; i < 3U; ++i) {
            for (auto j = 0U; j < 3U; ++j) {
                stiffness_matrix_terms(i_elem, 0, 1, i, j) = -a(i, j);
            }
        }
        for (auto i = 0U; i < 3U; ++i) {
            for (auto j = 0U; j < 3U; ++j) {
                stiffness_matrix_terms(i_elem, 1, 0, i, j) = -a(i, j);
            }
        }
        for (auto i = 0U; i < 3U; ++i) {
            for (auto j = 0U; j < 3U; ++j) {
                stiffness_matrix_terms(i_elem, 1, 1, i, j) = a(i, j);
            }
        }
    }
};

}  // namespace openturbine::springs
