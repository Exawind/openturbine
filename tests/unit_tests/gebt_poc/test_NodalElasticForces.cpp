#include <gtest/gtest.h>

#include "src/gebt_poc/NodalElasticForces.hpp"

#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {
struct NodalElasticForces_populate_strain {
    Kokkos::View<double[6]> sectional_strain;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        sectional_strain(0) = 1.1;
        sectional_strain(1) = 2.2;
        sectional_strain(2) = 3.3;
        sectional_strain(3) = 1.;
        sectional_strain(4) = 1.;
        sectional_strain(5) = 1.;
    }
};

struct NodalElasticForces_populate_position_derivatives {
    Kokkos::View<double[7]> position_vector_derivatives;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        position_vector_derivatives(0) = 1.;
        position_vector_derivatives(1) = 2.;
        position_vector_derivatives(2) = 3.;
        position_vector_derivatives(3) = 1.;
        position_vector_derivatives(4) = 0.;
        position_vector_derivatives(5) = 0.;
        position_vector_derivatives(6) = 0.;
    }
};

struct NodalElasticForces_populate_coords_derivatives {
    Kokkos::View<double[7]> gen_coords_derivatives;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        gen_coords_derivatives(0) = 0.1;
        gen_coords_derivatives(1) = 0.2;
        gen_coords_derivatives(2) = 0.3;
        gen_coords_derivatives(3) = 1.;
        gen_coords_derivatives(4) = 0.;
        gen_coords_derivatives(5) = 0.;
        gen_coords_derivatives(6) = 0.;
    }
};

TEST(SolverTest, NodalElasticForces) {
    auto sectional_strain = Kokkos::View<double*>("sectional_strain", 6);
    Kokkos::parallel_for(1, NodalElasticForces_populate_strain{sectional_strain});

    auto rotation = gen_alpha_solver::create_matrix({
        {1., 2., 3.},  // row 1
        {4., 5., 6.},  // row 2
        {7., 8., 9.}   // row 3
    });

    auto position_vector_derivatives = Kokkos::View<double*>("position_vector_derivatives", 7);
    Kokkos::parallel_for(
        1, NodalElasticForces_populate_position_derivatives{position_vector_derivatives}
    );

    auto gen_coords_derivatives = Kokkos::View<double*>("gen_coords_derivatives", 7);
    Kokkos::parallel_for(1, NodalElasticForces_populate_coords_derivatives{gen_coords_derivatives});

    auto stiffness = gen_alpha_solver::create_matrix({
        {1., 2., 3., 4., 5., 6.},       // row 1
        {2., 4., 6., 8., 10., 12.},     // row 2
        {3., 6., 9., 12., 15., 18.},    // row 3
        {4., 8., 12., 16., 20., 24.},   // row 4
        {5., 10., 15., 20., 25., 30.},  // row 5
        {6., 12., 18., 24., 30., 36.}   // row 6
    });
    auto elastic_forces_fc = Kokkos::View<double*>("elastic_forces_fc", 6);
    auto elastic_forces_fd = Kokkos::View<double*>("elastic_forces_fd", 6);
    NodalElasticForces(
        sectional_strain, rotation, position_vector_derivatives, gen_coords_derivatives, stiffness,
        elastic_forces_fc, elastic_forces_fd
    );

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        elastic_forces_fc, {-197.6, -395.2, -592.8, -790.4, -988., -1185.6}
    );
    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        elastic_forces_fd, {0., 0., 0., 0., 0., 0.}
    );
}
}