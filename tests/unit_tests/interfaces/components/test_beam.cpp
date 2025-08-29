#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include <gtest/gtest.h>

#include "interfaces/components/beam.hpp"
#include "interfaces/components/beam_input.hpp"
#include "model/model.hpp"

namespace openturbine::tests {

TEST(BeamComponentTest, InitialBeamHasCorrectRotation) {
    auto model = openturbine::Model();
    auto beam_input = openturbine::interfaces::components::BeamInput{};

    beam_input.element_order = 2UL;

    beam_input.ref_axis.coordinate_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.coordinates =
        std::vector{std::array{0., 0., 0.}, std::array{0.5, 0., 0.}, std::array{1., 0., 0.}};
    beam_input.ref_axis.twist_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.twist = std::vector{0., 0., 0.};

    auto mass_stiff_array =
        std::array{std::array{1., 0., 0., 0., 0., 0.}, std::array{0., 1., 0., 0., 0., 0.},
                   std::array{0., 0., 1., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0.},
                   std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 1.}};
    beam_input.sections = std::vector{
        openturbine::interfaces::components::Section(0., mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(0.5, mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(1., mass_stiff_array, mass_stiff_array)
    };

    auto beam = openturbine::interfaces::components::Beam(beam_input, model);
    const auto& beam_node_ids = model.GetBeamElement(0).node_ids;

    const auto& root_node = model.GetNode(beam_node_ids[0]);
    const auto root_rotation_quaternion =
        std::array{root_node.x0[3], root_node.x0[4], root_node.x0[5], root_node.x0[6]};

    const auto root_rotation_matrix = math::QuaternionToRotationMatrix(root_rotation_quaternion);

    EXPECT_NEAR(root_node.x0[0], 0., 1.e-16);
    EXPECT_NEAR(root_node.x0[1], 0., 1.e-16);
    EXPECT_NEAR(root_node.x0[2], 0., 1.e-16);

    EXPECT_NEAR(root_rotation_matrix[0][0], 1., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[0][1], 0., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[0][2], 0., 1.e-16);

    EXPECT_NEAR(root_rotation_matrix[1][0], 0., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[1][1], 1., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[1][2], 0., 1.e-16);

    EXPECT_NEAR(root_rotation_matrix[2][0], 0., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[2][1], 0., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[2][2], 1., 1.e-16);

    const auto& middle_node = model.GetNode(beam_node_ids[1]);
    const auto middle_rotation_quaternion =
        std::array{middle_node.x0[3], middle_node.x0[4], middle_node.x0[5], middle_node.x0[6]};

    const auto middle_rotation_matrix = math::QuaternionToRotationMatrix(middle_rotation_quaternion);

    EXPECT_NEAR(middle_node.x0[0], .5, 1.e-16);
    EXPECT_NEAR(middle_node.x0[1], 0., 1.e-16);
    EXPECT_NEAR(middle_node.x0[2], 0., 1.e-16);

    EXPECT_NEAR(middle_rotation_matrix[0][0], 1., 1.e-16);
    EXPECT_NEAR(middle_rotation_matrix[0][1], 0., 1.e-16);
    EXPECT_NEAR(middle_rotation_matrix[0][2], 0., 1.e-16);

    EXPECT_NEAR(middle_rotation_matrix[1][0], 0., 1.e-16);
    EXPECT_NEAR(middle_rotation_matrix[1][1], 1., 1.e-16);
    EXPECT_NEAR(middle_rotation_matrix[1][2], 0., 1.e-16);

    EXPECT_NEAR(middle_rotation_matrix[2][0], 0., 1.e-16);
    EXPECT_NEAR(middle_rotation_matrix[2][1], 0., 1.e-16);
    EXPECT_NEAR(middle_rotation_matrix[2][2], 1., 1.e-16);

    const auto& end_node = model.GetNode(beam_node_ids[2]);
    const auto end_rotation_quaternion =
        std::array{end_node.x0[3], end_node.x0[4], end_node.x0[5], end_node.x0[6]};

    const auto end_rotation_matrix = math::QuaternionToRotationMatrix(end_rotation_quaternion);

    EXPECT_NEAR(end_node.x0[0], 1., 1.e-16);
    EXPECT_NEAR(end_node.x0[1], 0., 1.e-16);
    EXPECT_NEAR(end_node.x0[2], 0., 1.e-16);

    EXPECT_NEAR(end_rotation_matrix[0][0], 1., 1.e-16);
    EXPECT_NEAR(end_rotation_matrix[0][1], 0., 1.e-16);
    EXPECT_NEAR(end_rotation_matrix[0][2], 0., 1.e-16);

    EXPECT_NEAR(end_rotation_matrix[1][0], 0., 1.e-16);
    EXPECT_NEAR(end_rotation_matrix[1][1], 1., 1.e-16);
    EXPECT_NEAR(end_rotation_matrix[1][2], 0., 1.e-16);

    EXPECT_NEAR(end_rotation_matrix[2][0], 0., 1.e-16);
    EXPECT_NEAR(end_rotation_matrix[2][1], 0., 1.e-16);
    EXPECT_NEAR(end_rotation_matrix[2][2], 1., 1.e-16);
}

TEST(BeamComponentTest, UnrotatedBeamHasIdentityRotationMatrix) {
    auto model = openturbine::Model();
    auto beam_input = openturbine::interfaces::components::BeamInput{};

    beam_input.element_order = 2UL;

    beam_input.ref_axis.coordinate_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.coordinates =
        std::vector{std::array{0., 0., 0.}, std::array{0.5, 0., 0.}, std::array{1., 0., 0.}};
    beam_input.ref_axis.twist_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.twist = std::vector{0., 0., 0.};

    auto mass_stiff_array =
        std::array{std::array{1., 0., 0., 0., 0., 0.}, std::array{0., 1., 0., 0., 0., 0.},
                   std::array{0., 0., 1., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0.},
                   std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 1.}};
    beam_input.sections = std::vector{
        openturbine::interfaces::components::Section(0., mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(0.5, mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(1., mass_stiff_array, mass_stiff_array)
    };

    auto beam = openturbine::interfaces::components::Beam(beam_input, model);
    const auto& beam_node_ids = model.GetBeamElement(0).node_ids;

    const auto& root_node = model.GetNode(beam_node_ids[0]);
    const auto root_rotation_quaternion =
        std::array{root_node.x0[3], root_node.x0[4], root_node.x0[5], root_node.x0[6]};

    const auto root_rotation_matrix = math::QuaternionToRotationMatrix(root_rotation_quaternion);

    EXPECT_NEAR(root_rotation_matrix[0][0], 1., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[0][1], 0., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[0][2], 0., 1.e-16);

    EXPECT_NEAR(root_rotation_matrix[1][0], 0., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[1][1], 1., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[1][2], 0., 1.e-16);

    EXPECT_NEAR(root_rotation_matrix[2][0], 0., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[2][1], 0., 1.e-16);
    EXPECT_NEAR(root_rotation_matrix[2][2], 1., 1.e-16);
}

TEST(BeamComponentTest, RotatedBeamAboutYAxisPointsAlongZAxis) {
    auto model = openturbine::Model();
    auto beam_input = openturbine::interfaces::components::BeamInput{};

    beam_input.element_order = 2UL;

    beam_input.ref_axis.coordinate_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.coordinates =
        std::vector{std::array{0., 0., 0.}, std::array{0.5, 0., 0.}, std::array{1., 0., 0.}};
    beam_input.ref_axis.twist_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.twist = std::vector{0., 0., 0.};

    auto mass_stiff_array =
        std::array{std::array{1., 0., 0., 0., 0., 0.}, std::array{0., 1., 0., 0., 0., 0.},
                   std::array{0., 0., 1., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0.},
                   std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 1.}};
    beam_input.sections = std::vector{
        openturbine::interfaces::components::Section(0., mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(0.5, mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(1., mass_stiff_array, mass_stiff_array)
    };

    beam_input.root.position =
        std::array{0., 0., 0., std::cos(std::numbers::pi / 4.), 0., std::sin(std::numbers::pi / 4.),
                   0.};

    auto beam = openturbine::interfaces::components::Beam(beam_input, model);
    const auto& beam_node_ids = model.GetBeamElement(0).node_ids;

    const auto& root_node = model.GetNode(beam_node_ids[0]);
    const auto root_rotation_quaternion =
        std::array{root_node.x0[3], root_node.x0[4], root_node.x0[5], root_node.x0[6]};

    const auto root_rotation_matrix = math::QuaternionToRotationMatrix(root_rotation_quaternion);

    EXPECT_NEAR(root_node.x0[0], 0., 1.e-15);
    EXPECT_NEAR(root_node.x0[1], 0., 1.e-15);
    EXPECT_NEAR(root_node.x0[2], 0., 1.e-15);

    EXPECT_NEAR(root_rotation_matrix[0][0], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[0][1], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[0][2], 1., 1.e-15);

    EXPECT_NEAR(root_rotation_matrix[1][0], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[1][1], 1., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[1][2], 0., 1.e-15);

    EXPECT_NEAR(root_rotation_matrix[2][0], -1., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[2][1], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[2][2], 0., 1.e-15);

    const auto& middle_node = model.GetNode(beam_node_ids[1]);
    const auto middle_rotation_quaternion =
        std::array{middle_node.x0[3], middle_node.x0[4], middle_node.x0[5], middle_node.x0[6]};

    const auto middle_rotation_matrix = math::QuaternionToRotationMatrix(middle_rotation_quaternion);

    EXPECT_NEAR(middle_node.x0[0], 0., 1.e-15);
    EXPECT_NEAR(middle_node.x0[1], 0., 1.e-15);
    EXPECT_NEAR(middle_node.x0[2], -.5, 1.e-15);

    EXPECT_NEAR(middle_rotation_matrix[0][0], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[0][1], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[0][2], 1., 1.e-15);

    EXPECT_NEAR(middle_rotation_matrix[1][0], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[1][1], 1., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[1][2], 0., 1.e-15);

    EXPECT_NEAR(middle_rotation_matrix[2][0], -1., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[2][1], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[2][2], 0., 1.e-15);

    const auto& end_node = model.GetNode(beam_node_ids[2]);
    const auto end_rotation_quaternion =
        std::array{end_node.x0[3], end_node.x0[4], end_node.x0[5], end_node.x0[6]};

    const auto end_rotation_matrix = math::QuaternionToRotationMatrix(end_rotation_quaternion);

    EXPECT_NEAR(end_node.x0[0], 0., 1.e-15);
    EXPECT_NEAR(end_node.x0[1], 0., 1.e-15);
    EXPECT_NEAR(end_node.x0[2], -1., 1.e-15);

    EXPECT_NEAR(end_rotation_matrix[0][0], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[0][1], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[0][2], 1., 1.e-15);

    EXPECT_NEAR(end_rotation_matrix[1][0], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[1][1], 1., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[1][2], 0., 1.e-15);

    EXPECT_NEAR(end_rotation_matrix[2][0], -1., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[2][1], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[2][2], 0., 1.e-15);
}

TEST(BeamComponentTest, RotatedBeamAboutZAxisPointsAlongYAxis) {
    auto model = openturbine::Model();
    auto beam_input = openturbine::interfaces::components::BeamInput{};

    beam_input.element_order = 2UL;

    beam_input.ref_axis.coordinate_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.coordinates =
        std::vector{std::array{0., 0., 0.}, std::array{0.5, 0., 0.}, std::array{1., 0., 0.}};
    beam_input.ref_axis.twist_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.twist = std::vector{0., 0., 0.};

    auto mass_stiff_array =
        std::array{std::array{1., 0., 0., 0., 0., 0.}, std::array{0., 1., 0., 0., 0., 0.},
                   std::array{0., 0., 1., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0.},
                   std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 1.}};
    beam_input.sections = std::vector{
        openturbine::interfaces::components::Section(0., mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(0.5, mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(1., mass_stiff_array, mass_stiff_array)
    };

    beam_input.root.position = std::array{
        0., 0., 0., std::cos(std::numbers::pi / 4.), 0., 0., std::sin(std::numbers::pi / 4.)
    };

    auto beam = openturbine::interfaces::components::Beam(beam_input, model);
    const auto& beam_node_ids = model.GetBeamElement(0).node_ids;

    const auto& root_node = model.GetNode(beam_node_ids[0]);
    const auto root_rotation_quaternion =
        std::array{root_node.x0[3], root_node.x0[4], root_node.x0[5], root_node.x0[6]};

    const auto root_rotation_matrix = math::QuaternionToRotationMatrix(root_rotation_quaternion);

    EXPECT_NEAR(root_node.x0[0], 0., 1.e-15);
    EXPECT_NEAR(root_node.x0[1], 0., 1.e-15);
    EXPECT_NEAR(root_node.x0[2], 0., 1.e-15);

    EXPECT_NEAR(root_rotation_matrix[0][0], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[0][1], -1., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[0][2], 0., 1.e-15);

    EXPECT_NEAR(root_rotation_matrix[1][0], 1., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[1][1], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[1][2], 0., 1.e-15);

    EXPECT_NEAR(root_rotation_matrix[2][0], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[2][1], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[2][2], 1., 1.e-15);

    const auto& middle_node = model.GetNode(beam_node_ids[1]);
    const auto middle_rotation_quaternion =
        std::array{middle_node.x0[3], middle_node.x0[4], middle_node.x0[5], middle_node.x0[6]};

    const auto middle_rotation_matrix = math::QuaternionToRotationMatrix(middle_rotation_quaternion);

    EXPECT_NEAR(middle_node.x0[0], 0., 1.e-15);
    EXPECT_NEAR(middle_node.x0[1], .5, 1.e-15);
    EXPECT_NEAR(middle_node.x0[2], 0., 1.e-15);

    EXPECT_NEAR(middle_rotation_matrix[0][0], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[0][1], -1., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[0][2], 0., 1.e-15);

    EXPECT_NEAR(middle_rotation_matrix[1][0], 1., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[1][1], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[1][2], 0., 1.e-15);

    EXPECT_NEAR(middle_rotation_matrix[2][0], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[2][1], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[2][2], 1., 1.e-15);

    const auto& end_node = model.GetNode(beam_node_ids[2]);
    const auto end_rotation_quaternion =
        std::array{end_node.x0[3], end_node.x0[4], end_node.x0[5], end_node.x0[6]};

    const auto end_rotation_matrix = math::QuaternionToRotationMatrix(end_rotation_quaternion);

    EXPECT_NEAR(end_node.x0[0], 0., 1.e-15);
    EXPECT_NEAR(end_node.x0[1], 1., 1.e-15);
    EXPECT_NEAR(end_node.x0[2], 0., 1.e-15);

    EXPECT_NEAR(end_rotation_matrix[0][0], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[0][1], -1., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[0][2], 0., 1.e-15);

    EXPECT_NEAR(end_rotation_matrix[1][0], 1., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[1][1], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[1][2], 0., 1.e-15);

    EXPECT_NEAR(end_rotation_matrix[2][0], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[2][1], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[2][2], 1., 1.e-15);
}

TEST(BeamComponentTest, RotatedBeamAboutXAxisStillPointsAlongXAxis) {
    auto model = openturbine::Model();
    auto beam_input = openturbine::interfaces::components::BeamInput{};

    beam_input.element_order = 2UL;

    beam_input.ref_axis.coordinate_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.coordinates =
        std::vector{std::array{0., 0., 0.}, std::array{0.5, 0., 0.}, std::array{1., 0., 0.}};
    beam_input.ref_axis.twist_grid = std::vector{0., 0.5, 1.0};
    beam_input.ref_axis.twist = std::vector{0., 0., 0.};

    auto mass_stiff_array =
        std::array{std::array{1., 0., 0., 0., 0., 0.}, std::array{0., 1., 0., 0., 0., 0.},
                   std::array{0., 0., 1., 0., 0., 0.}, std::array{0., 0., 0., 1., 0., 0.},
                   std::array{0., 0., 0., 0., 1., 0.}, std::array{0., 0., 0., 0., 0., 1.}};
    beam_input.sections = std::vector{
        openturbine::interfaces::components::Section(0., mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(0.5, mass_stiff_array, mass_stiff_array),
        openturbine::interfaces::components::Section(1., mass_stiff_array, mass_stiff_array)
    };

    beam_input.root.position =
        std::array{0., 0., 0., std::cos(std::numbers::pi / 4.), std::sin(std::numbers::pi / 4.),
                   0., 0.};

    auto beam = openturbine::interfaces::components::Beam(beam_input, model);
    const auto& beam_node_ids = model.GetBeamElement(0).node_ids;

    const auto& root_node = model.GetNode(beam_node_ids[0]);
    const auto root_rotation_quaternion =
        std::array{root_node.x0[3], root_node.x0[4], root_node.x0[5], root_node.x0[6]};

    const auto root_rotation_matrix = math::QuaternionToRotationMatrix(root_rotation_quaternion);

    EXPECT_NEAR(root_node.x0[0], 0., 1.e-15);
    EXPECT_NEAR(root_node.x0[1], 0., 1.e-15);
    EXPECT_NEAR(root_node.x0[2], 0., 1.e-15);

    EXPECT_NEAR(root_rotation_matrix[0][0], 1., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[0][1], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[0][2], 0., 1.e-15);

    EXPECT_NEAR(root_rotation_matrix[1][0], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[1][1], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[1][2], -1., 1.e-15);

    EXPECT_NEAR(root_rotation_matrix[2][0], 0., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[2][1], 1., 1.e-15);
    EXPECT_NEAR(root_rotation_matrix[2][2], 0., 1.e-15);

    const auto& middle_node = model.GetNode(beam_node_ids[1]);
    const auto middle_rotation_quaternion =
        std::array{middle_node.x0[3], middle_node.x0[4], middle_node.x0[5], middle_node.x0[6]};

    const auto middle_rotation_matrix = math::QuaternionToRotationMatrix(middle_rotation_quaternion);

    EXPECT_NEAR(middle_node.x0[0], .5, 1.e-15);
    EXPECT_NEAR(middle_node.x0[1], 0., 1.e-15);
    EXPECT_NEAR(middle_node.x0[2], 0., 1.e-15);

    EXPECT_NEAR(middle_rotation_matrix[0][0], 1., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[0][1], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[0][2], 0., 1.e-15);

    EXPECT_NEAR(middle_rotation_matrix[1][0], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[1][1], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[1][2], -1., 1.e-15);

    EXPECT_NEAR(middle_rotation_matrix[2][0], 0., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[2][1], 1., 1.e-15);
    EXPECT_NEAR(middle_rotation_matrix[2][2], 0., 1.e-15);

    const auto& end_node = model.GetNode(beam_node_ids[2]);
    const auto end_rotation_quaternion =
        std::array{end_node.x0[3], end_node.x0[4], end_node.x0[5], end_node.x0[6]};

    const auto end_rotation_matrix = math::QuaternionToRotationMatrix(end_rotation_quaternion);

    EXPECT_NEAR(end_node.x0[0], 1., 1.e-15);
    EXPECT_NEAR(end_node.x0[1], 0., 1.e-15);
    EXPECT_NEAR(end_node.x0[2], 0., 1.e-15);

    EXPECT_NEAR(end_rotation_matrix[0][0], 1., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[0][1], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[0][2], 0., 1.e-15);

    EXPECT_NEAR(end_rotation_matrix[1][0], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[1][1], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[1][2], -1., 1.e-15);

    EXPECT_NEAR(end_rotation_matrix[2][0], 0., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[2][1], 1., 1.e-15);
    EXPECT_NEAR(end_rotation_matrix[2][2], 0., 1.e-15);
}
}  // namespace openturbine::tests
