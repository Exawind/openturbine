#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "src/gebt_poc/mesh.h" 

TEST(MeshTest, Create1DMesh_1Element_2Node) {
    int numberOfElements = 1;
    int nodesPerElement = 2;
    auto mesh = openturbine::create1DMesh(numberOfElements, nodesPerElement);

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
    auto mesh = openturbine::create1DMesh(numberOfElements, nodesPerElement);

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
    auto mesh = openturbine::create1DMesh(numberOfElements, nodesPerElement);

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
    auto mesh = openturbine::create1DMesh(numberOfElements, nodesPerElement);

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
