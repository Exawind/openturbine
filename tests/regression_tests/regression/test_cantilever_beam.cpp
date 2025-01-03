#include <fstream>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"
#include "src/dof_management/compute_node_freedom_map_table.hpp"
#include "src/dof_management/create_constraint_freedom_table.hpp"
#include "src/dof_management/create_element_freedom_table.hpp"
#include "src/elements/beams/beam_element.hpp"
#include "src/elements/beams/beam_node.hpp"
#include "src/elements/beams/beam_section.hpp"
#include "src/elements/beams/beams.hpp"
#include "src/elements/beams/beams_input.hpp"
#include "src/elements/beams/create_beams.hpp"
#include "src/elements/elements.hpp"
#include "src/elements/masses/create_masses.hpp"
#include "src/model/model.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/step/step.hpp"
#include "src/types.hpp"

namespace openturbine::tests {

template <typename T>
void WriteMatrixToFile(const std::vector<std::vector<T>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << "\n";
        return;
    }
    for (const auto& innerVector : data) {
        for (const auto& element : innerVector) {
            file << element << ",";
        }
        file << "\n";
    }
    file.close();
}

TEST(DynamicBeamTest, CantileverBeamSineLoad) {
    // Mass matrix for uniform composite beam section
    constexpr auto mass_matrix = std::array{
        std::array{8.538e-2, 0., 0., 0., 0., 0.},   std::array{0., 8.538e-2, 0., 0., 0., 0.},
        std::array{0., 0., 8.538e-2, 0., 0., 0.},   std::array{0., 0., 0., 1.4433e-2, 0., 0.},
        std::array{0., 0., 0., 0., 0.40972e-2, 0.}, std::array{0., 0., 0., 0., 0., 1.0336e-2},
    };

    // Stiffness matrix for uniform composite beam section
    constexpr auto stiffness_matrix = std::array{
        std::array{1368.17e3, 0., 0., 0., 0., 0.},
        std::array{0., 88.56e3, 0., 0., 0., 0.},
        std::array{0., 0., 38.78e3, 0., 0., 0.},
        std::array{0., 0., 0., 16.9600e3, 17.6100e3, -0.3510e3},
        std::array{0., 0., 0., 17.6100e3, 59.1200e3, -0.3700e3},
        std::array{0., 0., 0., -0.3510e3, -0.3700e3, 141.470e3},
    };

    // Node locations (GLL quadrature)
    const auto node_s = std::vector{0., 0.17267316464601146, 0.5, 0.8273268353539885, 1.};

    // Element quadrature
    const auto quadrature = BeamQuadrature{
        {-0.9491079123427585, 0.1294849661688697},  {-0.7415311855993943, 0.27970539148927664},
        {-0.40584515137739696, 0.3818300505051189}, {6.123233995736766e-17, 0.4179591836734694},
        {0.4058451513773971, 0.3818300505051189},   {0.7415311855993945, 0.27970539148927664},
        {0.9491079123427585, 0.1294849661688697},
    };
    const auto sections = std::vector{
        BeamSection(0., mass_matrix, stiffness_matrix),
        BeamSection(1., mass_matrix, stiffness_matrix),
    };

    // Gravity vector
    constexpr auto gravity = std::array{0., 0., 0.};

    // Create model for managing nodes and constraints
    auto model = Model();

    // Build vector of nodes (straight along x axis, no rotation)
    std::vector<BeamNode> beam_nodes;
    std::transform(
        std::cbegin(node_s), std::cend(node_s), std::back_inserter(beam_nodes),
        [&](auto s) {
            return BeamNode(s, *model.AddNode({10 * s, 0., 0., 1., 0., 0., 0.}));
        }
    );

    // Define beam initialization
    const auto beams_input = BeamsInput({BeamElement(beam_nodes, sections, quadrature)}, gravity);
    const auto num_nodes = beam_nodes.size();

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // No Masses for this problem
    const auto masses_input = MassesInput({}, gravity);
    auto masses = CreateMasses(masses_input);

    // Create elements from beams
    auto elements = Elements{beams, masses};

    // Constraint inputs
    model.AddFixedBC(model.GetNode(0));

    // Solution parameters
    const bool is_dynamic_solve(true);
    const size_t max_iter(5);
    const double step_size(0.005);  // seconds
    const double rho_inf(0.0);

    // Create solver
    auto parameters = StepParameters(is_dynamic_solve, max_iter, step_size, rho_inf);
    auto constraints = Constraints(model.GetConstraints());
    auto state = model.CreateState();
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

    // First step
    Kokkos::deep_copy(
        Kokkos::subview(beams.node_FX, 0, num_nodes - 1, 2), 100. * std::sin(10.0 * 0.005)
    );
    auto converged = Step(parameters, solver, elements, state, constraints);
    EXPECT_EQ(converged, true);
    {
        const auto result = Kokkos::View<double[3]>("result");
        Kokkos::deep_copy(result, Kokkos::subview(state.q, 4, Kokkos::make_pair(0, 3)));
        expect_kokkos_view_1D_equal(result, {-8.15173937E-08, -1.86549248E-07, 6.97278045E-04});
    }
    // Second step
    Kokkos::deep_copy(
        Kokkos::subview(beams.node_FX, 0, num_nodes - 1, 2), 100. * std::sin(10.0 * 0.010)
    );
    converged = Step(parameters, solver, elements, state, constraints);
    EXPECT_EQ(converged, true);
    {
        const auto result = Kokkos::View<double[3]>("result");
        Kokkos::deep_copy(result, Kokkos::subview(state.q, 4, Kokkos::make_pair(0, 3)));
        expect_kokkos_view_1D_equal(result, {-1.00926258E-06, -7.91711079E-07, 2.65017558E-03});
    }

    // Third step
    Kokkos::deep_copy(
        Kokkos::subview(beams.node_FX, 0, num_nodes - 1, 2), 100. * std::sin(10.0 * 0.015)
    );
    converged = Step(parameters, solver, elements, state, constraints);
    EXPECT_EQ(converged, true);
    {
        const auto result = Kokkos::View<double[3]>("result");
        Kokkos::deep_copy(result, Kokkos::subview(state.q, 4, Kokkos::make_pair(0, 3)));
        expect_kokkos_view_1D_equal(result, {-5.05830945E-06, -2.29457246E-06, 6.30508154E-03});
    }
}

}  // namespace openturbine::tests
