#include "src/gen_alpha_poc/utilities.h"

#include <cmath>

namespace openturbine::gen_alpha_solver {

Kokkos::View<double*> create_vector(const std::vector<double>& values) {
    auto vector = Kokkos::View<double*>("vector", values.size());
    auto vector_host = Kokkos::create_mirror(vector);

    for (std::size_t index = 0; index < values.size(); ++index) {
        vector_host(index) = values[index];
    }
    Kokkos::deep_copy(vector, vector_host);

    return vector;
}

Kokkos::View<double**> create_matrix(const std::vector<std::vector<double>>& values) {
    auto matrix = Kokkos::View<double**>("matrix", values.size(), values.front().size());
    auto matrix_host = Kokkos::create_mirror(matrix);

    for (std::size_t row = 0; row < values.size(); ++row) {
        for (std::size_t column = 0; column < values.front().size(); ++column) {
            matrix_host(row, column) = values[row][column];
        }
    }
    Kokkos::deep_copy(matrix, matrix_host);

    return matrix;
}
}  // namespace openturbine::gen_alpha_solver
