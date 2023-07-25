#include <gtest/gtest.h>

#include "src/rigid_pendulum_poc/state.h"
#include "tests/unit_tests/rigid_pendulum_poc/test_utilities.h"

namespace openturbine::rigid_pendulum::tests {

TEST(StateTest, CreateDefaultState) {
    auto state = State();

    expect_kokkos_view_1D_equal(state.GetGeneralizedCoordinates(), {0.});
    expect_kokkos_view_1D_equal(state.GetVelocity(), {0.});
    expect_kokkos_view_1D_equal(state.GetAcceleration(), {0.});
    expect_kokkos_view_1D_equal(state.GetAlgorithmicAcceleration(), {0.});
}

TEST(StateTest, CreateStateWithGivenValues) {
    auto q = create_vector({1., 2., 3., 4.});
    auto v = create_vector({5., 6., 7.});
    auto state = State(q, v, v, v);

    expect_kokkos_view_1D_equal(state.GetGeneralizedCoordinates(), {1., 2., 3., 4.});
    expect_kokkos_view_1D_equal(state.GetVelocity(), {5., 6., 7.});
    expect_kokkos_view_1D_equal(state.GetAcceleration(), {5., 6., 7.});
    expect_kokkos_view_1D_equal(state.GetAlgorithmicAcceleration(), {5., 6., 7.});
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
    auto principal_moments_of_inertia = HostView1D("principal_moments_of_inertia", 3);
    principal_moments_of_inertia(0) = J.GetXComponent();
    principal_moments_of_inertia(1) = J.GetYComponent();
    principal_moments_of_inertia(2) = J.GetZComponent();
    expect_kokkos_view_1D_equal(principal_moments_of_inertia, {22., 29., 36.});
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
    auto mass_matrix = MassMatrix(15., Vector(0.234375, 0.46875, 0.234375));
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
    auto forces = HostView1D("forces", 3);
    forces(0) = f.GetXComponent();
    forces(1) = f.GetYComponent();
    forces(2) = f.GetZComponent();
    expect_kokkos_view_1D_equal(forces, {1., 2., 3.});

    auto m = generalized_forces.GetMoments();
    auto moments = HostView1D("moments", 3);
    moments(0) = m.GetXComponent();
    moments(1) = m.GetYComponent();
    moments(2) = m.GetZComponent();
    expect_kokkos_view_1D_equal(moments, {4., 5., 6.});
}

TEST(GeneralizedForcesTest, ExpectGeneralizedForcesToThrowWhenGiven1DVectorIsInvalid) {
    EXPECT_THROW(GeneralizedForces(create_vector({1., 2., 3., 4., 5.})), std::invalid_argument);
}

TEST(GeneralizedForcesTest, HeavyTopProblemFromBrulsAndCardona2010Paper) {
    auto mass = 15.;
    auto mass_matrix = MassMatrix(15., Vector(0.234375, 0.46875, 0.234375));

    auto gravity = Vector(0., 0., -9.81);
    auto forces = gravity * mass;

    auto angular_velocity = create_vector({0.3, 0.1, 0.8});
    auto J = mass_matrix.GetMomentOfInertiaMatrix();
    auto J_omega = multiply_matrix_with_vector(J, angular_velocity);

    auto angular_velocity_vector =
        Vector(angular_velocity(0), angular_velocity(1), angular_velocity(2));
    auto J_omega_vector = Vector(J_omega(0), J_omega(1), J_omega(2));
    auto moments = angular_velocity_vector.CrossProduct(J_omega_vector);

    auto generalized_forces = GeneralizedForces(forces, moments);
    expect_kokkos_view_1D_equal(
        generalized_forces.GetGeneralizedForces(), {0., 0., -147.15, -0.01875, 0., 0.00703125}
    );
}

}  // namespace openturbine::rigid_pendulum::tests
