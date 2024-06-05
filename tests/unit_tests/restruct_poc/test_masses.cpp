#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/restruct_poc/masses/create_masses.hpp"
#include "src/restruct_poc/masses/masses.hpp"
#include "src/restruct_poc/masses/masses_input.hpp"

namespace openturbine::restruct_poc::tests {

TEST(MassesTest, CreateMassesTest) {
    auto mass_input = MassesInput(
        {
            MassElement(MassNode({1., 2., 3.}, {1., 0., 0., 0.}), 1, {2, 3, 4}),
        },
        {0., 0., 0.}
    );

    auto masses = CreateMasses(mass_input, 2);

    expect_kokkos_view_2D_equal(
        Kokkos::subview(masses.node_Mstar, 0, Kokkos::ALL, Kokkos::ALL),
        {
            {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 2.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 3.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 4.0},
        }
    );

    expect_kokkos_view_2D_equal(masses.node_x0, {{1., 2., 3., 1., 0., 0., 0.}});
}

}  // namespace openturbine::restruct_poc::tests
