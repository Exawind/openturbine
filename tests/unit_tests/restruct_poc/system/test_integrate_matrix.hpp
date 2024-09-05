#include <array>
#include <string_view>

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine::tests {

template <size_t n_elem, size_t n_nodes>
auto get_node_state_indices() {
    using IndicesView = Kokkos::View<size_t[n_elem][n_nodes]>;
    auto indices = IndicesView("node_state_indices");
    const auto host_indices = Kokkos::create_mirror(indices);
    for (auto i = 0U; i < n_elem; ++i) {
        for (auto j = 0U; j < n_nodes; ++j) {
            host_indices(i, j) = i * n_nodes + j;
        }
    }
    Kokkos::deep_copy(indices, host_indices);
    return indices;
}

template <size_t n_qps>
auto get_qp_vector(std::string_view name, const std::array<double, n_qps>& vector_data) {
    using VectorView = Kokkos::View<double[n_qps]>;
    using HostVectorView = Kokkos::View<const double[n_qps], Kokkos::HostSpace>;
    auto vector = VectorView(std::string{name});
    const auto host_vector = Kokkos::create_mirror(vector);
    const auto vector_data_view = HostVectorView(vector_data.data());
    Kokkos::deep_copy(host_vector, vector_data_view);
    Kokkos::deep_copy(vector, host_vector);
    return vector;
}

template <size_t n_qps>
auto get_qp_weights(const std::array<double, n_qps>& weight_data) {
    return get_qp_vector<n_qps>("weights", weight_data);
}

template <size_t n_qps>
auto get_qp_jacobian(const std::array<double, n_qps>& jacobian_data) {
    return get_qp_vector<n_qps>("jacobian", jacobian_data);
}

template <size_t n_nodes, size_t n_qps>
auto get_shape_matrix(std::string_view name, const std::array<double, n_nodes * n_qps>& shape_data) {
    using ShapeView = Kokkos::View<double[n_nodes][n_qps], Kokkos::LayoutLeft>;
    using HostShapeView = Kokkos::View<const double[n_nodes][n_qps], Kokkos::HostSpace>;
    auto shape = ShapeView(std::string{name});
    const auto host_shape = Kokkos::create_mirror(shape);
    const auto shape_data_view = HostShapeView(shape_data.data());
    Kokkos::deep_copy(host_shape, shape_data_view);
    Kokkos::deep_copy(shape, host_shape);
    return shape;
}

template <size_t n_nodes, size_t n_qps>
auto get_shape_interp(const std::array<double, n_nodes * n_qps>& shape_data) {
    return get_shape_matrix<n_nodes, n_qps>("shape_interp", shape_data);
}

template <size_t n_nodes, size_t n_qps>
auto get_shape_interp_deriv(const std::array<double, n_nodes * n_qps>& shape_data) {
    return get_shape_matrix<n_nodes, n_qps>("shape_interp_deriv", shape_data);
}

template <size_t n_qps>
auto get_qp_matrix(std::string_view name, const std::array<double, n_qps * 6 * 6>& matrix_data) {
    using MatrixView = Kokkos::View<double[n_qps][6][6]>;
    using HostMatrixView = Kokkos::View<const double[n_qps][6][6], Kokkos::HostSpace>;
    auto matrix = MatrixView(std::string{name});
    const auto host_matrix = Kokkos::create_mirror(matrix);
    const auto matrix_data_view = HostMatrixView(matrix_data.data());
    Kokkos::deep_copy(host_matrix, matrix_data_view);
    Kokkos::deep_copy(matrix, host_matrix);
    return matrix;
}

template <size_t n_qps>
auto get_qp_M(const std::array<double, n_qps * 6 * 6>& M_data) {
    return get_qp_matrix<n_qps>("M", M_data);
}

template <size_t n_qps>
auto get_qp_Muu(const std::array<double, n_qps * 6 * 6>& M_data) {
    return get_qp_matrix<n_qps>("Muu", M_data);
}

template <size_t n_qps>
auto get_qp_Guu(const std::array<double, n_qps * 6 * 6>& M_data) {
    return get_qp_matrix<n_qps>("Guu", M_data);
}

template <size_t n_qps>
auto get_qp_Kuu(const std::array<double, n_qps * 6 * 6>& Puu_data) {
    return get_qp_matrix<n_qps>("Kuu", Puu_data);
}

template <size_t n_qps>
auto get_qp_Puu(const std::array<double, n_qps * 6 * 6>& Puu_data) {
    return get_qp_matrix<n_qps>("Puu", Puu_data);
}

template <size_t n_qps>
auto get_qp_Quu(const std::array<double, n_qps * 6 * 6>& Quu_data) {
    return get_qp_matrix<n_qps>("Quu", Quu_data);
}

template <size_t n_qps>
auto get_qp_Cuu(const std::array<double, n_qps * 6 * 6>& Cuu_data) {
    return get_qp_matrix<n_qps>("Cuu", Cuu_data);
}

template <size_t n_qps>
auto get_qp_Ouu(const std::array<double, n_qps * 6 * 6>& Ouu_data) {
    return get_qp_matrix<n_qps>("Ouu", Ouu_data);
}

}  // namespace openturbine::tests
