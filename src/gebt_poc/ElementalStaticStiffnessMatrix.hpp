#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/CalculateSectionalStrain.hpp"
#include "src/gebt_poc/InterpolateNodalValueDerivatives.hpp"
#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/NodalCurvature.hpp"
#include "src/gebt_poc/NodalElasticForces.hpp"
#include "src/gebt_poc/NodalStaticStiffnessMatrixComponents.hpp"
#include "src/gebt_poc/SectionalStiffness.hpp"
#include "src/gebt_poc/element.h"
#include "src/gebt_poc/interpolation.h"
#include "src/gebt_poc/quadrature.h"
#include "src/gebt_poc/section.h"
#include "src/gebt_poc/types.hpp"
#include "src/gen_alpha_poc/quaternion.h"
#include "src/gen_alpha_poc/utilities.h"

namespace openturbine::gebt_poc {
inline void ElementalStaticStiffnessMatrix(
    LieGroupFieldView::const_type position_vectors, LieGroupFieldView::const_type gen_coords,
    View2D_6x6::const_type stiffness, Quadrature quadrature, View2D stiffness_matrix
) {
    const auto n_nodes = gen_coords.extent(0);
    const auto order = n_nodes - 1;
    const auto n_quad_pts = quadrature.points.extent(0);

    auto nodes = VectorFieldView("nodes", n_nodes);
    Kokkos::deep_copy(
        nodes, Kokkos::subview(position_vectors, Kokkos::ALL, Kokkos::make_pair(0, 3))
    );

    // Allocate Views for some required intermediate variables
    auto gen_coords_qp = View1D_LieGroup("gen_coords_qp");
    auto gen_coords_derivatives_qp = View1D_LieGroup("gen_coords_derivatives_qp");
    auto position_vector_qp = View1D_LieGroup("position_vector_qp");
    auto pos_vector_derivatives_qp = View1D_LieGroup("pos_vector_derivatives_qp");
    auto curvature = View1D_Vector("curvature");
    auto sectional_strain = View1D_LieAlgebra("sectional_strain");
    auto sectional_stiffness = View2D_6x6("sectional_stiffness");
    auto O_matrix = View2D_6x6("O_matrix");
    auto P_matrix = View2D_6x6("P_matrix");
    auto Q_matrix = View2D_6x6("Q_matrix");

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

    Kokkos::deep_copy(stiffness_matrix, 0.);
    for (size_t i = 0; i < n_nodes; ++i) {
        for (size_t j = 0; j < n_nodes; ++j) {
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
                        member.team_barrier();
                        Kokkos::parallel_for(
                            Kokkos::ThreadVectorRange(member, LieGroupComponents),
                            [&](std::size_t component) {
                                gen_coords_qp(component) = gen_coords_interp(component);
                                position_vector_qp(component) = position_interp(component);
                                gen_coords_derivatives_qp(component) =
                                    gen_coord_deriv_interp(component);
                                pos_vector_derivatives_qp(component) =
                                    position_deriv_interp(component);
                            }
                        );
                    }
                );

                // Calculate the curvature and sectional strain
                NodalCurvature(gen_coords_qp, gen_coords_derivatives_qp, curvature);
                CalculateSectionalStrain(
                    pos_vector_derivatives_qp, gen_coords_derivatives_qp, curvature, sectional_strain
                );

                // Calculate the sectional stiffness matrix in inertial basis
                auto rotation_0 = gen_alpha_solver::EulerParameterToRotationMatrix(
                    Kokkos::subview(position_vector_qp, Kokkos::make_pair(3, 7))
                );
                auto rotation = gen_alpha_solver::EulerParameterToRotationMatrix(
                    Kokkos::subview(gen_coords_qp, Kokkos::make_pair(3, 7))
                );
                SectionalStiffness(stiffness, rotation_0, rotation, sectional_stiffness);

                // Calculate elastic forces i.e. F^C and F^D vectors
                auto elastic_forces_fc = View1D_LieAlgebra("elastic_forces_fc");
                auto elastic_forces_fd = View1D_LieAlgebra("elastic_forces_fd");
                NodalElasticForces(
                    sectional_strain, rotation, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                    sectional_stiffness, elastic_forces_fc, elastic_forces_fd
                );

                // Calculate the stiffness matrix components, i.e. O, P, and Q matrices
                NodalStaticStiffnessMatrixComponents(
                    elastic_forces_fc, pos_vector_derivatives_qp, gen_coords_derivatives_qp,
                    sectional_stiffness, O_matrix, P_matrix, Q_matrix
                );

                Kokkos::parallel_for(
                    Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
                        {0, 0}, {LieAlgebraComponents, LieAlgebraComponents}
                    ),
                    KOKKOS_LAMBDA(const size_t ii, const size_t jj) {
                        stiffness_matrix(
                            i * LieAlgebraComponents + ii, j * LieAlgebraComponents + jj
                        ) += quadrature.weights(k) *
                             (shape_function_vector(i) * P_matrix(ii, jj) *
                                  shape_function_derivative_vector(j) +
                              shape_function_vector(i) * Q_matrix(ii, jj) *
                                  shape_function_vector(j) * jacobian +
                              shape_function_derivative_vector(i) * sectional_stiffness(ii, jj) *
                                  shape_function_derivative_vector(j) / jacobian +
                              shape_function_derivative_vector(i) * O_matrix(ii, jj) *
                                  shape_function_vector(j));
                    }
                );
            }
        }
    }
}
}  // namespace openturbine::gebt_poc