#include <gtest/gtest.h>

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::restruct_poc::tests {

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_1) {
    // 90 degree rotation about the x axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.707107;
            q(1) = 0.707107;
            q(2) = 0.;
            q(3) = 0.;
        }
    );

    // Convert quaternion to rotation matrix and compare to expected rotation matrix
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    QuaternionToRotationMatrix(q, R_from_q);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        R_from_q, {{1., 0., 0.}, {0., 0., -1.}, {0., 1., 0.}}
    );
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_2) {
    // 90 degree rotation about the y axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.707107;
            q(1) = 0.;
            q(2) = 0.707107;
            q(3) = 0.;
        }
    );

    // Convert quaternion to rotation matrix and compare to expected rotation matrix
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    QuaternionToRotationMatrix(q, R_from_q);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        R_from_q, {{0., 0., 1.}, {0., 1., 0.}, {-1., 0., 0.}}
    );
}

TEST(QuaternionTest, ConvertQuaternionToRotationMatrix_3) {
    // 90 degree rotation about the z axis
    auto q = Kokkos::View<double[4]>("q");
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(const int) {
            q(0) = 0.707107;
            q(1) = 0.;
            q(2) = 0.;
            q(3) = 0.707107;
        }
    );

    // Convert quaternion to rotation matrix and compare to expected rotation matrix
    auto R_from_q = Kokkos::View<double[3][3]>("R_from_q");
    QuaternionToRotationMatrix(q, R_from_q);
    gen_alpha_solver::tests::expect_kokkos_view_2D_equal(
        R_from_q, {{0., -1., 0.}, {1., 0., 0.}, {0., 0., 1.}}
    );
}

}  // namespace openturbine::restruct_poc::tests
