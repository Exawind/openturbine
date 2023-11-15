#include <gtest/gtest.h>

#include "src/gen_alpha_poc/state.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gen_alpha_solver::tests {

TEST(StateTest, CreateDefaultState) {
    auto state = State();

    expect_kokkos_view_2D_equal(
        state.GetGeneralizedCoordinates(),
        {
            {0., 0., 0., 0., 0., 0., 0.}  // 7 elements
        }
    );
    expect_kokkos_view_2D_equal(
        state.GetVelocity(),
        {
            {0., 0., 0., 0., 0., 0.}  // 6 elements
        }
    );
    expect_kokkos_view_2D_equal(
        state.GetAcceleration(),
        {
            {0., 0., 0., 0., 0., 0.}  // 6 elements
        }
    );
    expect_kokkos_view_2D_equal(
        state.GetAlgorithmicAcceleration(),
        {
            {0., 0., 0., 0., 0., 0.}  // 6 elements
        }
    );
}

TEST(StateTest, CreateStateWithGivenValues) {
    auto state = State(
        create_matrix({
            {1., 2., 3., 4., 0., 0., 0.},  // 7 elements
        }),
        create_matrix({
            {5., 6., 7., 0., 0., 0.},  // 6 elements
        }),
        create_matrix({
            {5., 6., 7., 0., 0., 0.},  // 6 elements
        }),
        create_matrix({
            {5., 6., 7., 0., 0., 0.},  // 6 elements
        })
    );

    expect_kokkos_view_2D_equal(
        state.GetGeneralizedCoordinates(),
        {
            {1., 2., 3., 4., 0., 0., 0.}  // row 1
        }
    );
    expect_kokkos_view_2D_equal(
        state.GetVelocity(),
        {
            {5., 6., 7., 0., 0., 0.}  // row 1
        }
    );
    expect_kokkos_view_2D_equal(
        state.GetAcceleration(),
        {
            {5., 6., 7., 0., 0., 0.}  // row 1
        }
    );
    expect_kokkos_view_2D_equal(
        state.GetAlgorithmicAcceleration(),
        {
            {5., 6., 7., 0., 0., 0.}  // row 1
        }
    );
}

TEST(MassMatrixTest, CreateMassMatrixWithDefaultValues) {
    auto mass_matrix = MassMatrix();
    expect_kokkos_view_2D_equal(
        mass_matrix.GetMassMatrix(),
        {
            {1., 0., 0., 0., 0., 0.},  // row 1
            {0., 1., 0., 0., 0., 0.},  // row 2
            {0., 0., 1., 0., 0., 0.},  // row 3
            {0., 0., 0., 1., 0., 0.},  // row 4
            {0., 0., 0., 0., 1., 0.},  // row 5
            {0., 0., 0., 0., 0., 1.}   // row 6
        }
    );
}

TEST(MassMatrixTest, CreateMassMatrixWithGivenMassAndMomentOfInertia) {
    auto mass_matrix = MassMatrix(1., 2.);
    expect_kokkos_view_2D_equal(
        mass_matrix.GetMassMatrix(),
        {
            {1., 0., 0., 0., 0., 0.},  // row 1
            {0., 1., 0., 0., 0., 0.},  // row 2
            {0., 0., 1., 0., 0., 0.},  // row 3
            {0., 0., 0., 2., 0., 0.},  // row 4
            {0., 0., 0., 0., 2., 0.},  // row 5
            {0., 0., 0., 0., 0., 2.}   // row 6
        }
    );
}

TEST(MassMatrixTest, CreateMassMatrixWithGivenMassAndPrincipalMomentsOfInertia) {
    auto mass_matrix = MassMatrix(1., Vector(1., 2., 3.));
    expect_kokkos_view_2D_equal(
        mass_matrix.GetMassMatrix(),
        {
            {1., 0., 0., 0., 0., 0.},  // row 1
            {0., 1., 0., 0., 0., 0.},  // row 2
            {0., 0., 1., 0., 0., 0.},  // row 3
            {0., 0., 0., 1., 0., 0.},  // row 4
            {0., 0., 0., 0., 2., 0.},  // row 5
            {0., 0., 0., 0., 0., 3.}   // row 6
        }
    );
}

TEST(MassMatrixTest, ExpectMassMatrixToThrowWhenMassIsZero) {
    EXPECT_THROW(MassMatrix(0., 1.), std::invalid_argument);
}

TEST(MassMatrixTest, ExpectMassMatrixToThrowWhenMomentOfInertiaIsZero) {
    EXPECT_THROW(MassMatrix(1., 0.), std::invalid_argument);
}

TEST(MassMatrixTest, CreateMassMatrixWithGiven2DVector) {
    auto mass_matrix = MassMatrix(create_matrix({
        {1., 2., 3., 4., 5., 6.},        // row 1
        {7., 8., 9., 10., 11., 12.},     // row 2
        {13., 14., 15., 16., 17., 18.},  // row 3
        {19., 20., 21., 22., 23., 24.},  // row 4
        {25., 26., 27., 28., 29., 30.},  // row 5
        {31., 32., 33., 34., 35., 36.}   // row 6
    }));
    expect_kokkos_view_2D_equal(
        mass_matrix.GetMassMatrix(),
        {
            {1., 2., 3., 4., 5., 6.},        // row 1
            {7., 8., 9., 10., 11., 12.},     // row 2
            {13., 14., 15., 16., 17., 18.},  // row 3
            {19., 20., 21., 22., 23., 24.},  // row 4
            {25., 26., 27., 28., 29., 30.},  // row 5
            {31., 32., 33., 34., 35., 36.}   // row 6
        }
    );

    EXPECT_NEAR(mass_matrix.GetMass(), 1., 1e-15);

    auto J = mass_matrix.GetPrincipalMomentsOfInertia();

    EXPECT_NEAR(J.GetXComponent(), 22., 1.e-15);
    EXPECT_NEAR(J.GetYComponent(), 29., 1.e-15);
    EXPECT_NEAR(J.GetZComponent(), 36., 1.e-15);
}

TEST(MassMatrixTest, GetMomentOfInertiaMatrixFromMassMatrix) {
    auto mass_matrix = MassMatrix(1., Vector(1., 2., 3.));
    expect_kokkos_view_2D_equal(
        mass_matrix.GetMomentOfInertiaMatrix(),
        {
            {1., 0., 0.},  // row 1
            {0., 2., 0.},  // row 2
            {0., 0., 3.}   // row 3
        }
    );
}

TEST(MassMatrixTest, ExpectMassMatrixToThrowWhenGiven2DVectorIsInvalid) {
    EXPECT_THROW(
        MassMatrix(create_matrix({
            {1., 2., 3., 4., 5., 6.},        // row 1
            {7., 8., 9., 10., 11., 12.},     // row 2
            {13., 14., 15., 16., 17., 18.},  // row 3
            {19., 20., 21., 22., 23., 24.},  // row 4
            {25., 26., 27., 28., 29., 30.},  // row 5
            {31., 32., 33., 34., 35., 36.},  // row 6
            {37., 38., 39., 40., 41., 42.}   // row 7
        })),
        std::invalid_argument
    );
}

TEST(MassMatrixTest, HeavyTopProblemFromBrulsAndCardona2010Paper) {
    auto mass = 15.;
    auto J = Vector(0.234375, 0.46875, 0.234375);
    auto mass_matrix = MassMatrix(mass, J);
    expect_kokkos_view_2D_equal(
        mass_matrix.GetMassMatrix(),
        {
            {15., 0., 0., 0., 0., 0.},       // row 1
            {0., 15., 0., 0., 0., 0.},       // row 2
            {0., 0., 15., 0., 0., 0.},       // row 3
            {0., 0., 0., 0.234375, 0., 0.},  // row 4
            {0., 0., 0., 0., 0.46875, 0.},   // row 5
            {0., 0., 0., 0., 0., 0.234375}   // row 6
        }
    );
}

TEST(GeneralizedForcesTest, CreateGeneralizedForcesWithDefaultValues) {
    auto generalized_forces = GeneralizedForces();

    expect_kokkos_view_1D_equal(generalized_forces.GetGeneralizedForces(), {0., 0., 0., 0., 0., 0.});
}

TEST(GeneralizedForcesTest, CreateGeneralizedForcesWithGivenValues) {
    auto forces = Vector{1., 2., 3.};
    auto moments = Vector{4., 5., 6.};
    auto generalized_forces = GeneralizedForces(forces, moments);

    expect_kokkos_view_1D_equal(generalized_forces.GetGeneralizedForces(), {1., 2., 3., 4., 5., 6.});
}

TEST(GeneralizedForcesTest, CreateGeneralizedForcesWithGiven1DVector) {
    auto generalized_forces = GeneralizedForces(create_vector({1., 2., 3., 4., 5., 6.}));

    expect_kokkos_view_1D_equal(generalized_forces.GetGeneralizedForces(), {1., 2., 3., 4., 5., 6.});

    auto f = generalized_forces.GetForces();
    EXPECT_NEAR(f.GetXComponent(), 1., 1.e-15);
    EXPECT_NEAR(f.GetYComponent(), 2., 1.e-15);
    EXPECT_NEAR(f.GetZComponent(), 3., 1.e-15);

    auto m = generalized_forces.GetMoments();
    EXPECT_NEAR(m.GetXComponent(), 4., 1.e-15);
    EXPECT_NEAR(m.GetYComponent(), 5., 1.e-15);
    EXPECT_NEAR(m.GetZComponent(), 6., 1.e-15);
}

TEST(GeneralizedForcesTest, ExpectGeneralizedForcesToThrowWhenGiven1DVectorIsInvalid) {
    EXPECT_THROW(GeneralizedForces(create_vector({1., 2., 3., 4., 5.})), std::invalid_argument);
}

TEST(GeneralizedForcesTest, HeavyTopProblemFromBrulsAndCardona2010Paper) {
    auto mass = 15.;
    auto mass_matrix = MassMatrix(mass, Vector(0.234375, 0.46875, 0.234375));

    auto gravity = Vector(0., 0., -9.81);
    auto forces = gravity * mass;

    auto angular_velocity = create_vector({0.3, 0.1, 0.8});
    auto J = mass_matrix.GetMomentOfInertiaMatrix();
    auto J_omega = multiply_matrix_with_vector(J, angular_velocity);

    auto angular_velocity_host = Kokkos::create_mirror(angular_velocity);
    Kokkos::deep_copy(angular_velocity_host, angular_velocity);

    auto J_omega_host = Kokkos::create_mirror(J_omega);
    Kokkos::deep_copy(J_omega_host, J_omega);

    auto angular_velocity_vector =
        Vector(angular_velocity_host(0), angular_velocity_host(1), angular_velocity_host(2));
    auto J_omega_vector = Vector(J_omega_host(0), J_omega_host(1), J_omega_host(2));

    auto moments = angular_velocity_vector.CrossProduct(J_omega_vector);

    auto generalized_forces = GeneralizedForces(forces, moments);
    expect_kokkos_view_1D_equal(
        generalized_forces.GetGeneralizedForces(),
        {
            0.,         // force x
            0.,         // force y
            -147.15,    // force z
            -0.01875,   // moment x
            0.,         // moment y
            0.00703125  // moment z
        }
    );
}

}  // namespace openturbine::gen_alpha_solver::tests
