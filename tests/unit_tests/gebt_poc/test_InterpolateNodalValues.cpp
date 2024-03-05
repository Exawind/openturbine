#include <gtest/gtest.h>

#include "src/gebt_poc/InterpolateNodalValues.hpp"
#include "src/gebt_poc/interpolation.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc {

struct CalculateInterpolatedValues_populate_coords {
    Kokkos::View<double[14]> generalized_coords;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        generalized_coords(0) = 1.;
        generalized_coords(1) = 2.;
        generalized_coords(2) = 3.;
        generalized_coords(3) = 0.8775825618903728;
        generalized_coords(4) = 0.479425538604203;
        generalized_coords(5) = 0.;
        generalized_coords(6) = 0.;
        // node 2
        generalized_coords(7) = 2.;
        generalized_coords(8) = 3.;
        generalized_coords(9) = 4.;
        generalized_coords(10) = 0.7316888688738209;
        generalized_coords(11) = 0.6816387600233341;
        generalized_coords(12) = 0.;
        generalized_coords(13) = 0.;
    }
};

TEST(SolverTest, CalculateInterpolatedValues) {
    auto generalized_coords = Kokkos::View<double[14]>("generalized_coords");
    Kokkos::parallel_for(1, CalculateInterpolatedValues_populate_coords{generalized_coords});
    auto quadrature_pt = 0.;
    auto shape_function = LagrangePolynomial(1, quadrature_pt);

    auto interpolated_values = Kokkos::View<double[7]>("interpolated_values");
    InterpolateNodalValues(generalized_coords, shape_function, interpolated_values);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        interpolated_values, {1.5, 2.5, 3.5, 0.8109631195052179, 0.5850972729404622, 0., 0.}
    );
}

struct CalculateInterpolatedValues2D_populate_coords {
    Kokkos::View<double[2][7]> generalized_coords;

    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        generalized_coords(0, 0) = 1.;
        generalized_coords(0, 1) = 2.;
        generalized_coords(0, 2) = 3.;
        generalized_coords(0, 3) = 0.8775825618903728;
        generalized_coords(0, 4) = 0.479425538604203;
        generalized_coords(0, 5) = 0.;
        generalized_coords(0, 6) = 0.;
        // node 2
        generalized_coords(1, 0) = 2.;
        generalized_coords(1, 1) = 3.;
        generalized_coords(1, 2) = 4.;
        generalized_coords(1, 3) = 0.7316888688738209;
        generalized_coords(1, 4) = 0.6816387600233341;
        generalized_coords(1, 5) = 0.;
        generalized_coords(1, 6) = 0.;
    }
};

TEST(SolverTest, CalculateInterpolatedValues_2D) {
    auto generalized_coords = Kokkos::View<double[2][7]>("generalized_coords");
    Kokkos::parallel_for(1, CalculateInterpolatedValues2D_populate_coords{generalized_coords});
    auto quadrature_pt = 0.;
    auto shape_function = LagrangePolynomial(1, quadrature_pt);

    auto interpolated_values = Kokkos::View<double[7]>("interpolated_values");
    InterpolateNodalValues(generalized_coords, shape_function, interpolated_values);

    openturbine::gen_alpha_solver::tests::expect_kokkos_view_1D_equal(
        interpolated_values, {1.5, 2.5, 3.5, 0.8109631195052179, 0.5850972729404622, 0., 0.}
    );
}

}  // namespace openturbine::gebt_poc