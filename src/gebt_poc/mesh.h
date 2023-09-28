#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {
  class Mesh {
  public:
    int GetNumberOfElements() {
      return number_of_elements_;
    }

    int GetNumberOfNodes() {
      return number_of_nodes_;
    }

    void SetNumberOfElements(int number_of_elements) { number_of_elements_ = number_of_elements; }
    void SetNumberOfNodes(int number_of_nodes) { number_of_nodes_ = number_of_nodes; }

    void InitializeElementNodeConnectivity(int number_of_elements, int nodes_per_element) {
      connectivity_ = Kokkos::View<int**>("connectivity", number_of_elements, nodes_per_element);
    }

    KOKKOS_FUNCTION
    Kokkos::View<const int*> GetNodesForElement(int elementID) {
      return Kokkos::subview(connectivity_, elementID, Kokkos::ALL);
    }

    friend Mesh create1DMesh(int number_of_elements, int nodes_per_element);
  protected:
    int number_of_elements_;
    int number_of_nodes_;
    Kokkos::View<int**> connectivity_;
  };

  
  inline Mesh create1DMesh(int number_of_elements, int nodes_per_element) {
    Mesh mesh;
    mesh.SetNumberOfElements(number_of_elements);
    mesh.SetNumberOfNodes(number_of_elements * nodes_per_element - (number_of_elements - 1));
    mesh.InitializeElementNodeConnectivity(number_of_elements, nodes_per_element);

    auto set_element_connectivity = KOKKOS_LAMBDA(int elementID) {
      for(int node = 0; node < nodes_per_element; ++node) {
        mesh.connectivity_(elementID, node) = node + (nodes_per_element-1)*(elementID);
      }
    };

    Kokkos::parallel_for(number_of_elements, set_element_connectivity);
    return mesh;
  }
}
