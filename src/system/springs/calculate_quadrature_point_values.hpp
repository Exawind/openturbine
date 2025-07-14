#pragma once

#include <Kokkos_Core.hpp>

#include "system/springs/calculate_distance_components.hpp"
#include "system/springs/calculate_force_coefficients.hpp"
#include "system/springs/calculate_force_vectors.hpp"
#include "system/springs/calculate_length.hpp"
#include "system/springs/calculate_stiffness_matrix.hpp"

namespace openturbine::springs {

template <typename DeviceType>
struct CalculateQuadraturePointValues {
    typename Kokkos::View<double* [7], DeviceType>::const_type Q;

    typename Kokkos::View<size_t* [2], DeviceType>::const_type node_state_indices;
    typename Kokkos::View<double* [3], DeviceType>::const_type x0_;
    typename Kokkos::View<double*, DeviceType>::const_type l_ref_;
    typename Kokkos::View<double*, DeviceType>::const_type k_;

    Kokkos::View<double* [2][3], DeviceType> residual_vector_terms;
    Kokkos::View<double* [2][2][3][3], DeviceType> stiffness_matrix_terms;

    KOKKOS_FUNCTION
    void operator()(size_t element) const {
        const auto index_0 = node_state_indices(element, 0);
        const auto index_1 = node_state_indices(element, 1);

        const auto x0_data =
            Kokkos::Array<double, 3>{x0_(element, 0), x0_(element, 1), x0_(element, 2)};
        const auto u1_data = Kokkos::Array<double, 3>{Q(index_0, 0), Q(index_0, 1), Q(index_0, 2)};
        const auto u2_data = Kokkos::Array<double, 3>{Q(index_1, 0), Q(index_1, 1), Q(index_1, 2)};
        auto r_data = Kokkos::Array<double, 3>{};
        auto f_data = Kokkos::Array<double, 3>{};
        auto a_data = Kokkos::Array<double, 9>{};

        const auto x0 = typename Kokkos::View<double[3], DeviceType>::const_type(x0_data.data());
        const auto u1 = typename Kokkos::View<double[3], DeviceType>::const_type(u1_data.data());
        const auto u2 = typename Kokkos::View<double[3], DeviceType>::const_type(u2_data.data());
        const auto r = Kokkos::View<double[3], DeviceType>(r_data.data());
        const auto f = Kokkos::View<double[3], DeviceType>(f_data.data());
        const auto a = Kokkos::View<double[3][3], DeviceType>(a_data.data());

        const auto l_ref = l_ref_(element);
        const auto k = k_(element);

        springs::CalculateDistanceComponents(x0, u1, u2, r);
        const auto l = springs::CalculateLength<DeviceType>(r);
        const auto c1 = springs::CalculateForceCoefficient1<DeviceType>(k, l_ref, l);
        const auto c2 = springs::CalculateForceCoefficient2<DeviceType>(k, l_ref, l);
        springs::CalculateForceVectors(r, c1, f);
        springs::CalculateStiffnessMatrix(c1, c2, r, l, a);

        for (auto component = 0U; component < 3U; ++component) {
            residual_vector_terms(element, 0, component) = f(component);
            residual_vector_terms(element, 1, component) = -f(component);
        }

        for (auto component_1 = 0U; component_1 < 3U; ++component_1) {
            for (auto component_2 = 0U; component_2 < 3U; ++component_2) {
                stiffness_matrix_terms(element, 0, 0, component_1, component_2) =
                    a(component_1, component_2);
            }
        }
        for (auto component_1 = 0U; component_1 < 3U; ++component_1) {
            for (auto component_2 = 0U; component_2 < 3U; ++component_2) {
                stiffness_matrix_terms(element, 0, 1, component_1, component_2) =
                    -a(component_1, component_2);
            }
        }
        for (auto component_1 = 0U; component_1 < 3U; ++component_1) {
            for (auto component_2 = 0U; component_2 < 3U; ++component_2) {
                stiffness_matrix_terms(element, 1, 0, component_1, component_2) =
                    -a(component_1, component_2);
            }
        }
        for (auto component_1 = 0U; component_1 < 3U; ++component_1) {
            for (auto component_2 = 0U; component_2 < 3U; ++component_2) {
                stiffness_matrix_terms(element, 1, 1, component_1, component_2) =
                    a(component_1, component_2);
            }
        }
    }
};

}  // namespace openturbine::springs
