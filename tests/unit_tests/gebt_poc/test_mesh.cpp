#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/gebt_poc/mesh.h"

TEST(MeshTest, Create1DMesh_1Element_2Node) {
    int numberOfElements = 1;
    int nodesPerElement = 2;
    auto mesh = openturbine::gebt_poc::create1DMesh(numberOfElements, nodesPerElement);

    EXPECT_EQ(mesh.GetNumberOfElements(), numberOfElements);
    EXPECT_EQ(mesh.GetNumberOfNodes(), 2);

    auto nodeList = mesh.GetNodesForElement(0);
    EXPECT_EQ(nodeList.extent(0), nodesPerElement);

    auto hostNodeList = Kokkos::create_mirror(nodeList);
    Kokkos::deep_copy(hostNodeList, nodeList);
    EXPECT_EQ(hostNodeList(0), 0);
    EXPECT_EQ(hostNodeList(1), 1);
}

TEST(MeshTest, Create1DMesh_1Element_5Node) {
    int numberOfElements = 1;
    int nodesPerElement = 5;
    auto mesh = openturbine::gebt_poc::create1DMesh(numberOfElements, nodesPerElement);

    EXPECT_EQ(mesh.GetNumberOfElements(), numberOfElements);
    EXPECT_EQ(mesh.GetNumberOfNodes(), 5);

    auto nodeList = mesh.GetNodesForElement(0);
    EXPECT_EQ(nodeList.extent(0), nodesPerElement);

    auto hostNodeList = Kokkos::create_mirror(nodeList);
    Kokkos::deep_copy(hostNodeList, nodeList);
    EXPECT_EQ(hostNodeList(0), 0);
    EXPECT_EQ(hostNodeList(1), 1);
    EXPECT_EQ(hostNodeList(2), 2);
    EXPECT_EQ(hostNodeList(3), 3);
    EXPECT_EQ(hostNodeList(4), 4);
}

TEST(MeshTest, Create1DMesh_2Element_5Node) {
    int numberOfElements = 2;
    int nodesPerElement = 5;
    auto mesh = openturbine::gebt_poc::create1DMesh(numberOfElements, nodesPerElement);

    EXPECT_EQ(mesh.GetNumberOfElements(), numberOfElements);
    EXPECT_EQ(mesh.GetNumberOfNodes(), 9);

    {
        auto nodeList = mesh.GetNodesForElement(0);
        EXPECT_EQ(nodeList.extent(0), nodesPerElement);

        auto hostNodeList = Kokkos::create_mirror(nodeList);
        Kokkos::deep_copy(hostNodeList, nodeList);
        EXPECT_EQ(hostNodeList(0), 0);
        EXPECT_EQ(hostNodeList(1), 1);
        EXPECT_EQ(hostNodeList(2), 2);
        EXPECT_EQ(hostNodeList(3), 3);
        EXPECT_EQ(hostNodeList(4), 4);
    }

    {
        auto nodeList = mesh.GetNodesForElement(1);
        EXPECT_EQ(nodeList.extent(0), nodesPerElement);

        auto hostNodeList = Kokkos::create_mirror(nodeList);
        Kokkos::deep_copy(hostNodeList, nodeList);
        EXPECT_EQ(hostNodeList(0), 4);
        EXPECT_EQ(hostNodeList(1), 5);
        EXPECT_EQ(hostNodeList(2), 6);
        EXPECT_EQ(hostNodeList(3), 7);
        EXPECT_EQ(hostNodeList(4), 8);
    }
}

TEST(MeshTest, Create1DMesh_3Element_3Node) {
    int numberOfElements = 3;
    int nodesPerElement = 3;
    auto mesh = openturbine::gebt_poc::create1DMesh(numberOfElements, nodesPerElement);

    EXPECT_EQ(mesh.GetNumberOfElements(), numberOfElements);
    EXPECT_EQ(mesh.GetNumberOfNodes(), 7);

    {
        auto nodeList = mesh.GetNodesForElement(0);
        EXPECT_EQ(nodeList.extent(0), nodesPerElement);

        auto hostNodeList = Kokkos::create_mirror(nodeList);
        Kokkos::deep_copy(hostNodeList, nodeList);
        EXPECT_EQ(hostNodeList(0), 0);
        EXPECT_EQ(hostNodeList(1), 1);
        EXPECT_EQ(hostNodeList(2), 2);
    }

    {
        auto nodeList = mesh.GetNodesForElement(1);
        EXPECT_EQ(nodeList.extent(0), nodesPerElement);

        auto hostNodeList = Kokkos::create_mirror(nodeList);
        Kokkos::deep_copy(hostNodeList, nodeList);
        EXPECT_EQ(hostNodeList(0), 2);
        EXPECT_EQ(hostNodeList(1), 3);
        EXPECT_EQ(hostNodeList(2), 4);
    }

    {
        auto nodeList = mesh.GetNodesForElement(2);
        EXPECT_EQ(nodeList.extent(0), nodesPerElement);

        auto hostNodeList = Kokkos::create_mirror(nodeList);
        Kokkos::deep_copy(hostNodeList, nodeList);
        EXPECT_EQ(hostNodeList(0), 4);
        EXPECT_EQ(hostNodeList(1), 5);
        EXPECT_EQ(hostNodeList(2), 6);
    }
}

namespace openturbine::gebt_poc::impl {
class TestMesh : public openturbine::gebt_poc::Mesh {
public:
    void TestSetNumberOfElements(int number_of_elements) { SetNumberOfElements(number_of_elements); }
    void TestSetNumberOfNodes(int number_of_nodes) { SetNumberOfNodes(number_of_nodes); }

    void TestInitializeConnectivity(int number_of_elements, int nodes_per_element) {
        InitializeElementNodeConnectivity(number_of_elements, nodes_per_element);
    }

    void TestCheckConsistency() { CheckConsistency(); }

    Kokkos::View<int**> TestGetConnectivity() { return connectivity_; }
};
}  // namespace openturbine::gebt_poc::impl

TEST(MeshTest, CheckConsistency_WrongNumberOfElements) {
    openturbine::gebt_poc::impl::TestMesh mesh;

    mesh.TestSetNumberOfElements(2);
    mesh.TestSetNumberOfNodes(5);

    mesh.TestInitializeConnectivity(1, 5);

    EXPECT_THROW(mesh.TestCheckConsistency(), std::domain_error);
}

TEST(MeshTest, CheckConsistency_InvalidNode) {
    openturbine::gebt_poc::impl::TestMesh mesh;

    mesh.TestSetNumberOfElements(1);
    mesh.TestSetNumberOfNodes(2);

    mesh.TestInitializeConnectivity(1, 2);

    auto connectivity = mesh.TestGetConnectivity();
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(int) {
            connectivity(0, 0) = 0;
            connectivity(0, 1) = 2;
        }
    );

    EXPECT_THROW(mesh.TestCheckConsistency(), std::domain_error);
}

TEST(MeshTest, CheckConsistency_NonUniqueNode) {
    openturbine::gebt_poc::impl::TestMesh mesh;

    mesh.TestSetNumberOfElements(1);
    mesh.TestSetNumberOfNodes(3);

    mesh.TestInitializeConnectivity(1, 3);

    auto connectivity = mesh.TestGetConnectivity();
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(int) {
            connectivity(0, 0) = 0;
            connectivity(0, 1) = 1;
            connectivity(0, 2) = 0;
        }
    );

    EXPECT_THROW(mesh.TestCheckConsistency(), std::domain_error);
}

TEST(MeshTest, CheckConsistency_NonUniqueNode_2Element) {
    openturbine::gebt_poc::impl::TestMesh mesh;

    mesh.TestSetNumberOfElements(2);
    mesh.TestSetNumberOfNodes(5);

    mesh.TestInitializeConnectivity(2, 3);

    auto connectivity = mesh.TestGetConnectivity();
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(int) {
            connectivity(0, 0) = 0;
            connectivity(0, 1) = 1;
            connectivity(0, 2) = 2;
            connectivity(1, 0) = 2;
            connectivity(1, 1) = 3;
            connectivity(1, 2) = 3;
        }
    );

    EXPECT_THROW(mesh.TestCheckConsistency(), std::domain_error);
}

TEST(MeshTest, CheckConsistency_Pass_Bar) {
    openturbine::gebt_poc::impl::TestMesh mesh;

    mesh.TestSetNumberOfElements(2);
    mesh.TestSetNumberOfNodes(5);

    mesh.TestInitializeConnectivity(2, 3);

    auto connectivity = mesh.TestGetConnectivity();
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(int) {
            connectivity(0, 0) = 0;
            connectivity(0, 1) = 1;
            connectivity(0, 2) = 2;
            connectivity(1, 0) = 2;
            connectivity(1, 1) = 3;
            connectivity(1, 2) = 4;
        }
    );

    EXPECT_NO_THROW(mesh.TestCheckConsistency());
}

TEST(MeshTest, CheckConsistency_Pass_Plus) {
    openturbine::gebt_poc::impl::TestMesh mesh;

    mesh.TestSetNumberOfElements(4);
    mesh.TestSetNumberOfNodes(9);

    mesh.TestInitializeConnectivity(4, 3);

    auto connectivity = mesh.TestGetConnectivity();
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(int) {
            connectivity(0, 0) = 0;
            connectivity(0, 1) = 1;
            connectivity(0, 2) = 2;
            connectivity(1, 0) = 2;
            connectivity(1, 1) = 3;
            connectivity(1, 2) = 4;
            connectivity(2, 0) = 2;
            connectivity(2, 1) = 5;
            connectivity(2, 2) = 6;
            connectivity(3, 0) = 2;
            connectivity(3, 1) = 7;
            connectivity(3, 2) = 8;
        }
    );

    EXPECT_NO_THROW(mesh.TestCheckConsistency());
}
