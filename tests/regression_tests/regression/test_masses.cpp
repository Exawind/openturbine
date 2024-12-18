#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/elements/elements.hpp"
#include "src/elements/masses/create_masses.hpp"
#include "src/model/model.hpp"
#include "src/state/state.hpp"
#include "src/step/update_system_variables.hpp"
#include "src/types.hpp"

namespace openturbine::tests {

inline auto SetUpMasses() {
    auto model = Model();
    constexpr auto mass_matrix = std::array{
        std::array{1., 0., 0., 0., 0., 0.}, std::array{0., 1., 0., 0., 0., 0.},
        std::array{0., 0., 1., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0.},
        std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 1.},
    };
    model.AddNode(
        {0, 0, 0, 1, 0, 0, 0},  // Initial position and orientation
        {0, 0, 0, 1, 0, 0, 0},  // Initial displacement
        {0, 0, 0, 0, 0, 0},     // Initial velocity
        {0, 0, 0, 0, 0, 0}      // Initial acceleration
    );
    constexpr auto gravity = std::array{0., 0., 9.81};
    const auto masses_input = MassesInput(
        {
            MassElement(model.GetNode(0), mass_matrix),
        },
        gravity
    );

    auto masses = CreateMasses(masses_input);

    auto parameters = StepParameters(false, 0, 0., 0.);
    auto state = model.CreateState();
    UpdateSystemVariablesMasses(parameters, masses, state);

    return masses;
}

TEST(MassesTest, NodeInitialPosition) {
    const auto masses = SetUpMasses();
    expect_kokkos_view_2D_equal(
        masses.qp_x0,
        {
            {0., 0., 0., 1., 0., 0., 0.},
        }
    );
}

TEST(MassesTest, NodeInitialDisplacement) {
    const auto masses = SetUpMasses();
    expect_kokkos_view_2D_equal(
        masses.node_u,
        {
            {0., 0., 0., 1., 0., 0., 0.},
        }
    );
}

TEST(MassesTest, NodeInitialVelocity) {
    const auto masses = SetUpMasses();
    expect_kokkos_view_2D_equal(
        masses.node_u_dot,
        {
            {0., 0., 0., 0., 0., 0.},
        }
    );
}

TEST(MassesTest, NodeInitialAcceleration) {
    const auto masses = SetUpMasses();
    expect_kokkos_view_2D_equal(
        masses.node_u_ddot,
        {
            {0., 0., 0., 0., 0., 0.},
        }
    );
}

TEST(MassesTest, MassMatrixInMaterialFrame) {
    const auto masses = SetUpMasses();
    expect_kokkos_view_2D_equal(
        Kokkos::subview(masses.qp_Mstar, 0, Kokkos::ALL, Kokkos::ALL),
        {
            {1., 0., 0., 0., 0., 0.},
            {0., 1., 0., 0., 0., 0.},
            {0., 0., 1., 0., 0., 0.},
            {0., 0., 0., 1., 0., 0.},
            {0., 0., 0., 0., 1., 0.},
            {0., 0., 0., 0., 0., 1.},
        }
    );
}

TEST(MassesTest, GravityVector) {
    const auto masses = SetUpMasses();
    expect_kokkos_view_1D_equal(masses.gravity, {0., 0., 9.81});
}

}  // namespace openturbine::tests
