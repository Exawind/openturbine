#include <gtest/gtest.h>

#include "src/gebt_poc/NodalCurvature.hpp"
#include "src/gebt_poc/interpolation.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {
struct NodalCurvature_populate_coords {
    Kokkos::View<double[7]> gen_coords;
    gen_alpha_solver::Quaternion q;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        gen_coords(0) = 0.;
        gen_coords(1) = 0.;
        gen_coords(2) = 0.;
        gen_coords(3) = q.GetScalarComponent();
        gen_coords(4) = q.GetXComponent();
        gen_coords(5) = q.GetYComponent();
        gen_coords(6) = q.GetZComponent();
    }
};

struct NodalCurvature_populate_derivative {
    Kokkos::View<double[7]> gen_coords_derivative;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        gen_coords_derivative(0) = 0.;
        gen_coords_derivative(1) = 0.;
        gen_coords_derivative(2) = 0.;
        gen_coords_derivative(3) = -0.0257045;
        gen_coords_derivative(4) = -0.0230032;
        gen_coords_derivative(5) = 0.030486;
        gen_coords_derivative(6) = 0.0694527;
    }
};

TEST(SolverTest, NodalCurvature) {
    auto rotation_matrix = gen_alpha_solver::RotationMatrix(
        0.8146397707387071, -0.4884001129794905, 0.31277367787652416, 0.45607520213614394,
        0.8726197541000288, 0.17472886066955512, -0.3582700851693625, 0.00030723936311904954,
        0.933617936672551
    );
    auto q = gen_alpha_solver::rotation_matrix_to_quaternion(rotation_matrix);

    auto gen_coords = Kokkos::View<double[7]>("gen_coords");
    Kokkos::parallel_for(1, NodalCurvature_populate_coords{gen_coords, q});

    auto gen_coords_derivative = Kokkos::View<double[7]>("gen_coords_derivative");

    Kokkos::parallel_for(1, NodalCurvature_populate_derivative{gen_coords_derivative});

    auto curvature = Kokkos::View<double[3]>("curvature");
    NodalCurvature(gen_coords, gen_coords_derivative, curvature);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        curvature, {-0.03676700256944363, 0.062023963818612256, 0.15023478838786522}
    );
}
}  // namespace openturbine::gebt_poc