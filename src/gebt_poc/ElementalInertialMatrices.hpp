#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/InterpolateNodalValueDerivatives.hpp"
#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/NodalDynamicStiffnessMatrix.hpp"
#include "src/gebt_poc/NodalGyroscopicMatrix.hpp"
#include "src/gebt_poc/SectionalMassMatrix.hpp"
#include "src/gebt_poc/element.h"
#include "src/gebt_poc/interpolation.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {
inline void ElementalInertialMatrices(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    LieAlgebraFieldView::const_type velocity, LieAlgebraFieldView::const_type acceleration,
    View2D_6x6::const_type mass_matrix, Quadrature quadrature, View2D element_mass_matrix,
    View2D element_gyroscopic_matrix, View2D element_dynamic_stiffness_matrix
) {
    const auto n_nodes = gen_coords.extent(0);
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.points.extent(0);

    auto nodes = Kokkos::View<double* [3]>("nodes", n_nodes);
    Kokkos::deep_copy(
        nodes, Kokkos::subview(position_vectors, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );

    // Allocate Views for some required intermediate variables
    auto gen_coords_qp = View1D_LieGroup("gen_coords_qp");
    auto gen_coords_derivatives_qp = View1D_LieGroup("gen_coords_derivatives_qp");
    auto position_vector_qp = View1D_LieGroup("position_vector_qp");
    auto pos_vector_derivatives_qp = View1D_LieGroup("pos_vector_derivatives_qp");
    auto velocity_qp = View1D_LieAlgebra("velocity_qp");
    auto acceleration_qp = View1D_LieAlgebra("acceleration_qp");
    auto sectional_mass_matrix = View2D_6x6("sectional_mass_matrix");

    auto policy = Kokkos::TeamPolicy<>(1, Kokkos::AUTO(), Kokkos::AUTO());
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using unmanaged_memory = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using ScratchView1D = Kokkos::View<double*, scratch_space, unmanaged_memory>;
    using ScratchView2D = Kokkos::View<double**, scratch_space, unmanaged_memory>;
    const auto team_scratch_size =
        5 * ScratchView2D::shmem_size(n_quad_pts, n_nodes) + ScratchView1D::shmem_size(n_nodes);
    const auto thread_scratch_size = ScratchView1D::shmem_size(n_nodes);
    auto shape_functions = View2D("shape functions", n_quad_pts, n_nodes);
    auto shape_function_derivatives = View2D("shape function derivatives", n_quad_pts, n_nodes);
    policy.set_scratch_size(
        0, Kokkos::PerTeam(team_scratch_size), Kokkos::PerThread(thread_scratch_size)
    );
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {
            auto shapes = ComputeLagrangePolynomials(member, order, quadrature.points);
            auto derivatives =
                ComputeLagrangePolynomialDerivatives(member, order, quadrature.points);
            member.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorMDRange(member, shapes.extent(0), shapes.extent(1)),
                [&](std::size_t i, std::size_t j) {
                    shape_functions(i, j) = shapes(i, j);
                    shape_function_derivatives(i, j) = derivatives(i, j);
                }
            );
        }
    );

    Kokkos::deep_copy(element_mass_matrix, 0.);

    for (size_t k = 0; k < n_quad_pts; ++k) {
        // Calculate required interpolated values at the quadrature point
        auto shape_function_vector = View1D("sfv", n_nodes);
        auto shape_function_derivative_vector = View1D("sfv", n_nodes);
        Kokkos::parallel_for(
            n_nodes,
            KOKKOS_LAMBDA(std::size_t l) {
                shape_function_vector(l) = shape_functions(k, l);
                shape_function_derivative_vector(l) = shape_function_derivatives(k, l);
            }
        );

        auto jacobian = CalculateJacobian(nodes, shape_function_derivative_vector);
        Kokkos::parallel_for(
            policy,
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {
                auto gen_coords_interp =
                    InterpolateNodalValues(member, gen_coords, shape_function_vector);
                auto gen_coord_deriv_interp = InterpolateNodalValueDerivatives(
                    member, gen_coords, shape_function_derivative_vector, jacobian
                );
                auto position_interp =
                    InterpolateNodalValues(member, position_vectors, shape_function_vector);
                auto position_deriv_interp = InterpolateNodalValueDerivatives(
                    member, position_vectors, shape_function_derivative_vector, jacobian
                );
                auto velocity_interp =
                    InterpolateNodalValues(member, velocity, shape_function_vector);
                auto acceleration_interp =
                    InterpolateNodalValues(member, acceleration, shape_function_vector);
                member.team_barrier();
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(member, LieGroupComponents),
                    [&](std::size_t component) {
                        gen_coords_qp(component) = gen_coords_interp(component);
                        position_vector_qp(component) = position_interp(component);
                        gen_coords_derivatives_qp(component) = gen_coord_deriv_interp(component);
                        pos_vector_derivatives_qp(component) = position_deriv_interp(component);
                    }
                );
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(member, LieAlgebraComponents),
                    [&](std::size_t component) {
                        velocity_qp(component) = velocity_interp(component);
                        acceleration_qp(component) = acceleration_interp(component);
                    }
                );
            }
        );

        // Calculate the sectional mass matrix in inertial basis
        auto rotation_0 = gen_alpha_solver::EulerParameterToRotationMatrix(
            Kokkos::subview(position_vector_qp, Kokkos::make_pair(3, 7))
        );
        auto rotation = gen_alpha_solver::EulerParameterToRotationMatrix(
            Kokkos::subview(gen_coords_qp, Kokkos::make_pair(3, 7))
        );
        SectionalMassMatrix(mass_matrix, rotation_0, rotation, sectional_mass_matrix);

        // Calculate the gyroscopic matrix in inertial basis
        auto gyroscopic_matrix = View2D_6x6("gyroscopic_matrix");
        NodalGyroscopicMatrix(velocity_qp, sectional_mass_matrix, gyroscopic_matrix);

        // Calculate the dynamic stiffness matrix in inertial basis
        auto dynamic_stiffness_matrix = View2D_6x6("dynamic_stiffness_matrix");
        NodalDynamicStiffnessMatrix(
            velocity_qp, acceleration_qp, sectional_mass_matrix, dynamic_stiffness_matrix
        );

        Kokkos::parallel_for(
            policy,
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorMDRange(member, n_nodes, n_nodes, 6, 6),
                    [&](std::size_t i, std::size_t j, std::size_t m, std::size_t n) {
                        const auto a = quadrature.weights(k) * shape_function_vector(i) *
                                       shape_function_vector(j) * jacobian;
                        const auto mass_contribution = a * sectional_mass_matrix(m, n);
                        const auto gyroscopic_contribution = a * gyroscopic_matrix(m, n);
                        const auto stiffness_contribution = a * dynamic_stiffness_matrix(m, n);
                        Kokkos::atomic_add(
                            &element_mass_matrix(i * 6 + m, j * 6 + n), mass_contribution
                        );
                        Kokkos::atomic_add(
                            &element_gyroscopic_matrix(i * 6 + m, j * 6 + n), gyroscopic_contribution
                        );
                        Kokkos::atomic_add(
                            &element_dynamic_stiffness_matrix(i * 6 + m, j * 6 + n),
                            stiffness_contribution
                        );
                    }
                );
            }
        );
    }
}
}  // namespace openturbine::gebt_poc