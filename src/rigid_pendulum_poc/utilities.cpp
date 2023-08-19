#include "src/rigid_pendulum_poc/utilities.h"

#include <cmath>

namespace openturbine::rigid_pendulum {

KOKKOS_FUNCTION
bool close_to(double a, double b, double epsilon) {
    auto delta = std::abs(a - b);
    a = std::abs(a);
    b = std::abs(b);

    if (a < epsilon) {
        if (b < epsilon) {
            return true;
        }
        return false;
    }

    return (delta / a) < epsilon ? true : false;
}

double wrap_angle_to_pi(double angle) {
    double wrapped_angle = std::fmod(angle, 2. * kPI);

    // Check if the angle is close to PI or -PI to avoid numerical issues
    if (close_to(wrapped_angle, kPI)) {
        return kPI;
    }
    if (close_to(wrapped_angle, -kPI)) {
        return -kPI;
    }

    if (wrapped_angle > kPI) {
        wrapped_angle -= 2. * kPI;
    }
    if (wrapped_angle < -kPI) {
        wrapped_angle += 2. * kPI;
    }

    return wrapped_angle;
}

Kokkos::View<double*> create_identity_vector(size_t size) {
    auto vector = Kokkos::View<double*>("vector", size);

    Kokkos::parallel_for(
        size, KOKKOS_LAMBDA(int i) { vector(i) = 1.; }
    );

    return vector;
}

Kokkos::View<double**> create_identity_matrix(size_t size) {
    auto matrix = Kokkos::View<double**>("matrix", size, size);
    auto diagonal_entries = Kokkos::RangePolicy(0, size);
    auto fill_diagonal = KOKKOS_LAMBDA(int index) {
        matrix(index, index) = 1.;
    };

    Kokkos::parallel_for(diagonal_entries, fill_diagonal);

    return matrix;
}

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

Kokkos::View<double**> transpose_matrix(const Kokkos::View<double**> matrix) {
    auto transposed_matrix =
        Kokkos::View<double**>("transposed_matrix", matrix.extent(1), matrix.extent(0));
    auto entries = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
        {0, 0}, {matrix.extent(1), matrix.extent(0)}
    );
    auto transpose = KOKKOS_LAMBDA(int row, int column) {
        transposed_matrix(row, column) = matrix(column, row);
    };

    Kokkos::parallel_for(entries, transpose);

    return transposed_matrix;
}

Kokkos::View<double**> create_cross_product_matrix(const Kokkos::View<double*> vector) {
    if (vector.extent(0) != 3) {
        throw std::invalid_argument("The provided vector must have 3 elements");
    }

    auto matrix = Kokkos::View<double**>("cross_product_matrix", 3, 3);
    auto populate_matrix = KOKKOS_LAMBDA(int) {
        matrix(0, 0) = 0.;
        matrix(0, 1) = -vector(2);
        matrix(0, 2) = vector(1);
        matrix(1, 0) = vector(2);
        matrix(1, 1) = 0.;
        matrix(1, 2) = -vector(0);
        matrix(2, 0) = -vector(1);
        matrix(2, 1) = vector(0);
        matrix(2, 2) = 0.;
    };

    Kokkos::parallel_for(1, populate_matrix);

    return matrix;
}

Kokkos::View<double*> multiply_matrix_with_vector(
    const Kokkos::View<double**> matrix, const Kokkos::View<double*> vector
) {
    if (matrix.extent(1) != vector.extent(0)) {
        throw std::invalid_argument(
            "The number of columns of the matrix must be equal to the number of rows of the vector"
        );
    }

    auto result = Kokkos::View<double*>("result", matrix.extent(0));
    auto entries = Kokkos::RangePolicy(0, matrix.extent(0));
    auto multiply_row = KOKKOS_LAMBDA(int row) {
        double sum = 0.;
        for (size_t column = 0; column < matrix.extent(1); ++column) {
            sum += matrix(row, column) * vector(column);
        }
        result(row) = sum;
    };

    Kokkos::parallel_for(entries, multiply_row);

    return result;
}

Kokkos::View<double**> multiply_matrix_with_matrix(
    const Kokkos::View<double**> matrix_a, const Kokkos::View<double**> matrix_b
) {
    auto a_n_columns = matrix_a.extent(1);
    auto b_n_rows = matrix_b.extent(0);

    if (b_n_rows != a_n_columns) {
        throw std::invalid_argument(
            "The number of columns of the first matrix must be equal to the number of rows of the "
            "second matrix"
        );
    }

    auto n_rows = matrix_a.extent(0);
    auto n_columns = matrix_b.extent(1);
    auto result = Kokkos::View<double**>("result", n_rows, n_columns);
    auto entries = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
        {0, 0}, {n_rows, n_columns}
    );
    auto multiply_row_column = KOKKOS_LAMBDA(int row, int column) {
        double sum = 0.;
        for (size_t i = 0; i < matrix_a.extent(1); ++i) {
            sum += matrix_a(row, i) * matrix_b(i, column);
        }
        result(row, column) = sum;
    };

    Kokkos::parallel_for(entries, multiply_row_column);

    return result;
}

Kokkos::View<double**> multiply_matrix_with_scalar(
    const Kokkos::View<double**> matrix, double scalar
) {
    auto result = Kokkos::View<double**>("result", matrix.extent(0), matrix.extent(1));
    auto entries = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>(
        {0, 0}, {matrix.extent(0), matrix.extent(1)}
    );
    auto multiply_row_column = KOKKOS_LAMBDA(int row, int column) {
        result(row, column) = matrix(row, column) * scalar;
    };

    Kokkos::parallel_for(entries, multiply_row_column);

    return result;
}

}  // namespace openturbine::rigid_pendulum
