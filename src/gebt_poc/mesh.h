#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::gebt_poc {
class Mesh {
public:
    int GetNumberOfElements() const { return number_of_elements_; }

    int GetNumberOfNodes() const { return number_of_nodes_; }

    KOKKOS_FUNCTION
    auto GetNodesForElement(int element_id) const {
        return Kokkos::subview(connectivity_, element_id, Kokkos::ALL);
    }

    void CheckNodeIDConsistency() {
        auto range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {connectivity_.extent(0), connectivity_.extent(1)}
        );
        int max_node_id;
        auto connectivity = connectivity_;
        Kokkos::parallel_reduce(
            range_policy,
            KOKKOS_LAMBDA(int i, int j, int& local_max) {
                if (connectivity(i, j) > local_max) {
                    local_max = connectivity(i, j);
                }
            },
            Kokkos::Max<int>(max_node_id)
        );

        if (max_node_id >= number_of_nodes_) {
            throw std::domain_error("Connectivity references node ID above the expected range");
        }
    }

    friend Mesh Create1DMesh(int number_of_elements, int nodes_per_element);

protected:
    Mesh() = default;
    void SetNumberOfElements(int number_of_elements) { number_of_elements_ = number_of_elements; }
    void SetNumberOfNodes(int number_of_nodes) { number_of_nodes_ = number_of_nodes; }

    void InitializeElementNodeConnectivity(int number_of_elements, int nodes_per_element) {
        connectivity_ = Kokkos::View<int**>("connectivity", number_of_elements, nodes_per_element);
    }

    void CheckElementConsistency() {
        if (static_cast<std::size_t>(number_of_elements_) != connectivity_.extent(0)) {
            throw std::domain_error("Connectivity does not contain expected number of elements");
        }
    }

    void CheckNodeUniquenessConsistency() {
        for (std::size_t element = 0; element < connectivity_.extent(0); ++element) {
            auto node_list = GetNodesForElement(element);
            auto tmp_node_list = Kokkos::View<double*>("tmp", node_list.extent(0));
            auto host_node_list = Kokkos::create_mirror(tmp_node_list);
            Kokkos::deep_copy(tmp_node_list, node_list);
            Kokkos::deep_copy(host_node_list, tmp_node_list);
            auto node_list_vector = std::vector<int>(host_node_list.extent(0));
            for (std::size_t node = 0; node < node_list_vector.size(); ++node) {
                node_list_vector[node] = host_node_list(node);
            }
            std::sort(std::begin(node_list_vector), std::end(node_list_vector));
            auto first_duplicate =
                std::adjacent_find(std::begin(node_list_vector), std::end(node_list_vector));
            if (first_duplicate != end(node_list_vector)) {
                throw std::domain_error("Nodes within an element must be unique");
            }
        }
    }

    void CheckConsistency() {
        CheckElementConsistency();
        CheckNodeIDConsistency();
        CheckNodeUniquenessConsistency();
    }

    int number_of_elements_;
    int number_of_nodes_;
    Kokkos::View<int**> connectivity_;
};

inline Mesh Create1DMesh(int number_of_elements, int nodes_per_element) {
    Mesh mesh;
    mesh.SetNumberOfElements(number_of_elements);
    mesh.SetNumberOfNodes(number_of_elements * nodes_per_element - (number_of_elements - 1));
    mesh.InitializeElementNodeConnectivity(number_of_elements, nodes_per_element);

    auto set_element_connectivity = KOKKOS_LAMBDA(int element_id) {
        for (int node = 0; node < nodes_per_element; ++node) {
            mesh.connectivity_(element_id, node) = node + (nodes_per_element - 1) * (element_id);
        }
    };

    Kokkos::parallel_for(number_of_elements, set_element_connectivity);

    mesh.CheckConsistency();
    return mesh;
}
}  // namespace openturbine::gebt_poc
