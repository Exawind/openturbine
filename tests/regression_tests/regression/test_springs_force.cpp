#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"
#include "src/dof_management/compute_node_freedom_map_table.hpp"
#include "src/dof_management/create_constraint_freedom_table.hpp"
#include "src/dof_management/create_element_freedom_table.hpp"
#include "src/elements/beams/create_beams.hpp"
#include "src/elements/elements.hpp"
#include "src/elements/masses/create_masses.hpp"
#include "src/elements/springs/create_springs.hpp"
#include "src/elements/springs/springs.hpp"
#include "src/model/model.hpp"
#include "src/solver/solver.hpp"
#include "src/step/step.hpp"
#include "src/step/update_system_variables_springs.hpp"

namespace openturbine::tests {

inline auto SetUpSpringsForceTest() {
    auto model = Model();

    // Add two nodes for the spring element
    model.AddNode(
        {0, 0, 0, 1, 0, 0, 0},  // First node at origin
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}
    );
    model.AddNode(
        {1, 0, 0, 1, 0, 0, 0},  // Second node at (1,0,0)
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}
    );

    const auto springs_input = SpringsInput({SpringElement(
        std::array{model.GetNode(0), model.GetNode(1)},
        10.,  // Spring stiffness coeff.
        1.    // Undeformed length
    )});

    auto springs = CreateSprings(springs_input);
    auto state = model.CreateState();

    return std::make_tuple(springs, state);
}

TEST(SpringsForceTest, ZeroDisplacement) {
    auto [springs, state] = SetUpSpringsForceTest();

    UpdateSystemVariablesSprings(springs, state);

    auto l_ref_host = Kokkos::create_mirror(springs.l_ref);
    auto l_host = Kokkos::create_mirror(springs.l);
    auto c1_host = Kokkos::create_mirror(springs.c1);
    auto f_host = Kokkos::create_mirror(springs.f);
    auto a_host = Kokkos::create_mirror(springs.a);

    Kokkos::deep_copy(l_ref_host, springs.l_ref);
    Kokkos::deep_copy(l_host, springs.l);
    Kokkos::deep_copy(c1_host, springs.c1);
    Kokkos::deep_copy(f_host, springs.f);
    Kokkos::deep_copy(a_host, springs.a);

    EXPECT_DOUBLE_EQ(l_ref_host(0), 1.);  // Undeformed length = 1.
    EXPECT_DOUBLE_EQ(l_host(0), 1.);      // Current length = 1.
    EXPECT_DOUBLE_EQ(c1_host(0), 0.);     // c1 = 0.
    expect_kokkos_view_1D_equal(
        Kokkos::subview(springs.f, 0, Kokkos::ALL), {0., 0., 0.}
    );  // No force
    expect_kokkos_view_2D_equal(
        Kokkos::subview(springs.a, 0, Kokkos::ALL, Kokkos::ALL),
        {{-10., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}}
    );
}

TEST(SpringsForceTest, UnitDisplacement) {
    auto [springs, state] = SetUpSpringsForceTest();

    state.q(1, 0) = 1.;  // Displace second node in x direction
    UpdateSystemVariablesSprings(springs, state);

    auto l_ref_host = Kokkos::create_mirror(springs.l_ref);
    auto l_host = Kokkos::create_mirror(springs.l);
    auto c1_host = Kokkos::create_mirror(springs.c1);
    auto c2_host = Kokkos::create_mirror(springs.c2);
    auto f_host = Kokkos::create_mirror(springs.f);
    auto a_host = Kokkos::create_mirror(springs.a);

    Kokkos::deep_copy(l_ref_host, springs.l_ref);
    Kokkos::deep_copy(l_host, springs.l);
    Kokkos::deep_copy(c1_host, springs.c1);
    Kokkos::deep_copy(c2_host, springs.c2);
    Kokkos::deep_copy(f_host, springs.f);
    Kokkos::deep_copy(a_host, springs.a);

    EXPECT_DOUBLE_EQ(l_ref_host(0), 1.);                       // Undeformed length = 1.
    EXPECT_DOUBLE_EQ(l_host(0), 2.);                           // Current length = 2.
    EXPECT_DOUBLE_EQ(c1_host(0), -5.);                         // c1 = -5.
    EXPECT_DOUBLE_EQ(c2_host(0), 10. * 1. / std::pow(2., 3));  // c2 = 10. / 8.
    expect_kokkos_view_1D_equal(
        Kokkos::subview(springs.f, 0, Kokkos::ALL), {-10., 0., 0.}
    );  // Force = c1 * {r}
    expect_kokkos_view_2D_equal(
        Kokkos::subview(springs.a, 0, Kokkos::ALL, Kokkos::ALL),
        {{-10., 0., 0.}, {0., -5., 0.}, {0., 0., -5.}}
    );
}

inline auto SetUpSpringsForceTestUsingSolver() {
    auto model = Model();

    // Add two nodes for the spring element
    model.AddNode(
        {0, 0, 0, 1, 0, 0, 0},  // First node at origin
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}
    );
    model.AddNode(
        {2, 0, 0, 1, 0, 0, 0},  // Second node at (1,0,0)
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}
    );

    const auto springs_input = SpringsInput({SpringElement(
        std::array{model.GetNode(0), model.GetNode(1)},
        10.,     // Spring stiffness coeff.
        0.00000  // Undeformed length
    )});

    constexpr auto mass_matrix = std::array{
        std::array{1., 0., 0., 0., 0., 0.}, std::array{0., 1., 0., 0., 0., 0.},
        std::array{0., 0., 1., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0.},
        std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 1.},
    };

    // Create all element types (even if some are empty)
    const auto beams_input = BeamsInput({}, {0., 0., 0.});
    auto beams = CreateBeams(beams_input);
    const auto masses_input =
        MassesInput({MassElement(model.GetNode(1), mass_matrix)}, {0., 0., 0.});
    auto masses = CreateMasses(masses_input);
    auto springs = CreateSprings(springs_input);

    model.AddFixedBC(model.GetNode(0));

    // Set up step parameters
    constexpr bool is_dynamic_solve(true);
    constexpr size_t max_iter(6);
    constexpr double step_size(0.01);
    constexpr double rho_inf(0.0);
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);

    // Set up solver components
    auto state = model.CreateState();
    auto constraints = Constraints(model.GetConstraints());
    auto elements = Elements{beams, masses, springs};

    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);

    auto solver = Solver(
        state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
        constraints.type, constraints.base_node_freedom_table, constraints.target_node_freedom_table,
        constraints.row_range
    );

    for (auto time_step = 0U; time_step < 300; ++time_step) {
        auto converged = Step(parameters, solver, elements, state, constraints);
        EXPECT_TRUE(converged);
    }

    EXPECT_EQ(state.q(0, 0), 0.);
    EXPECT_EQ(state.q(1, 0), -3.9961930498908851);
}

TEST(SpringsForceTestUsingSolver, ZeroDisplacement) {
    SetUpSpringsForceTestUsingSolver();
}

}  // namespace openturbine::tests
