#include <iostream>

#include <gtest/gtest.h>

#include "elements/beams/beam_element.hpp"
#include "model/model.hpp"

namespace openturbine::tests {

TEST(ModelTest, AddNodeToModel) {
    Model model;
    ASSERT_EQ(model.NumNodes(), 0);

    // Add a node to the model
    constexpr auto pos = std::array{0., 0., 0.};
    constexpr auto rot = std::array{1., 0., 0., 0.};
    constexpr auto v = std::array{0., 0., 0.};
    constexpr auto omega = std::array{0., 0., 0.};
    auto node_id = model.AddNode()
                       .SetPosition(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3])
                       .SetVelocity(v[0], v[1], v[2], omega[0], omega[1], omega[2])
                       .Build();
    auto node = model.GetNode(node_id);
    ASSERT_EQ(node_id, 0);
    ASSERT_EQ(node.id, 0);
    ASSERT_EQ(model.NumNodes(), 1);

    auto nodes = model.GetNodes();
    ASSERT_EQ(nodes.size(), 1);
}

TEST(ModelTest, AddBeamElementToModel) {
    Model model;

    constexpr auto pos = std::array{0., 0., 0.};
    constexpr auto rot = std::array{1., 0., 0., 0.};
    constexpr auto v = std::array{0., 0., 0.};
    constexpr auto omega = std::array{0., 0., 0.};

    // Add couple of nodes to the model
    auto node1_id = model.AddNode()
                        .SetElemLocation(0.)
                        .SetPosition(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3])
                        .SetVelocity(v[0], v[1], v[2], omega[0], omega[1], omega[2])
                        .Build();

    auto node2_id = model.AddNode()
                        .SetElemLocation(1.)
                        .SetPosition(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3])
                        .SetVelocity(v[0], v[1], v[2], omega[0], omega[1], omega[2])
                        .Build();

    // Add a beam element to the model
    auto sections = std::vector<BeamSection>{};
    auto quadrature = BeamQuadrature{};
    model.AddBeamElement({node1_id, node2_id}, sections, quadrature);

    ASSERT_EQ(model.NumBeamElements(), 1);

    auto elements = model.GetBeamElements();
    ASSERT_EQ(elements.size(), 1);
}

TEST(ModelTest, AddMassElementToModel) {
    Model model;
    auto node_id = model.AddNode().Build();
    auto mass_matrix = std::array<std::array<double, 6>, 6>{};
    model.AddMassElement(node_id, mass_matrix);

    ASSERT_EQ(model.NumMassElements(), 1);

    auto elements = model.GetMassElements();
    ASSERT_EQ(elements.size(), 1);
}

TEST(ModelTest, ModelConstructorWithDefaults) {
    const Model model;
    ASSERT_EQ(model.NumNodes(), 0);
    ASSERT_EQ(model.NumBeamElements(), 0);
    ASSERT_EQ(model.NumMassElements(), 0);
    ASSERT_EQ(model.NumConstraints(), 0);
}

TEST(ModelTest, ModelCreateState) {
    Model model;

    // Rotation of 1 radian around x
    auto R1 = RotationVectorToQuaternion({1., 0., 0.});
    auto R2 = RotationVectorToQuaternion({0., 1., 0.});

    // Create node with initial position and displacement from initial position
    static_cast<void>(model.AddNode()
                          .SetPosition(1., 2., 3., R1[0], R1[1], R1[2], R1[3])
                          .SetDisplacement(3., 2., 1., R2[0], R2[1], R2[2], R2[3])
                          .Build());

    using DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
    // Create state object from model
    auto state = model.CreateState<DeviceType>();

    // Verify initial position
    const auto x0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.x0);
    const auto exact_x0 = std::array{1., 2., 3., R1[0], R1[1], R1[2], R1[3]};
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(x0(0, i), exact_x0[i], 1.e-15);
    }

    // Verify initial displacement
    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.q);
    const auto exact_q = std::array{3., 2., 1., R2[0], R2[1], R2[2], R2[3]};
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(q(0, i), exact_q[i], 1.e-15);
    }

    // Verify current position (initial position plus displacement)
    auto Rt = QuaternionCompose(R2, R1);
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.x);
    const auto exact_x = std::array{4., 4., 4., Rt[0], Rt[1], Rt[2], Rt[3]};
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(x(0, i), exact_x[i], 1.e-15);
    }
}

TEST(ModelTest, ModelCreateSystem) {
    Model model;

    // Rotation of 1 radian around x
    auto R1 = RotationVectorToQuaternion({1., 0., 0.});
    auto R2 = RotationVectorToQuaternion({0., 1., 0.});

    // Create node with initial position and displacement from initial position
    static_cast<void>(model.AddNode()
                          .SetPosition(1., 2., 3., R1[0], R1[1], R1[2], R1[3])
                          .SetDisplacement(3., 2., 1., R2[0], R2[1], R2[2], R2[3])
                          .Build());

    // Create state object from model
    auto [state, elements, constraints] = model.CreateSystem();

    // Verify initial position
    const auto x0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.x0);
    const auto exact_x0 = std::array{1., 2., 3., R1[0], R1[1], R1[2], R1[3]};
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(x0(0, i), exact_x0[i], 1.e-15);
    }

    // Verify initial displacement
    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.q);
    const auto exact_q = std::array{3., 2., 1., R2[0], R2[1], R2[2], R2[3]};
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(q(0, i), exact_q[i], 1.e-15);
    }

    // Verify current position (initial position plus displacement)
    auto Rt = QuaternionCompose(R2, R1);
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.x);
    const auto exact_x = std::array{4., 4., 4., Rt[0], Rt[1], Rt[2], Rt[3]};
    for (auto i = 0U; i < 7U; ++i) {
        EXPECT_NEAR(x(0, i), exact_x[i], 1.e-15);
    }

    EXPECT_EQ(elements.NumElementsInSystem(), 0);
    EXPECT_EQ(constraints.num_constraints, 0);
}

}  // namespace openturbine::tests
