#include "src/rigid_pendulum_poc/utilities.h"

#include <cmath>

namespace openturbine::rigid_pendulum {

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

HostView1D create_identity_vector(size_t size) {
    auto vector = HostView1D("vector", size);

    Kokkos::parallel_for(
        size, KOKKOS_LAMBDA(int i) { vector(i) = 1.; }
    );

    return vector;
}

HostView2D create_identity_matrix(size_t size) {
    auto matrix = HostView2D("matrix", size, size);
    auto diagonal_entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, size);
    auto fill_diagonal = [matrix](int index) {
        matrix(index, index) = 1.;
    };

    Kokkos::parallel_for(diagonal_entries, fill_diagonal);

    return matrix;
}

HostView1D create_vector(const std::vector<double>& values) {
    auto vector = HostView1D("vector", values.size());
    auto entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, values.size());
    auto fill_vector = [vector, values](int index) {
        vector(index) = values[index];
    };

    Kokkos::parallel_for(entries, fill_vector);

    return vector;
}

HostView2D create_matrix(const std::vector<std::vector<double>>& values) {
    auto matrix = HostView2D("matrix", values.size(), values.front().size());
    auto entries = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
        {0, 0}, {values.size(), values.front().size()}
    );
    auto fill_matrix = [matrix, values](int row, int column) {
        matrix(row, column) = values[row][column];
    };

    Kokkos::parallel_for(entries, fill_matrix);

    return matrix;
}

HostView2D create_cross_product_matrix(HostView1D vector) {
    if (vector.extent(0) != 3) {
        throw std::invalid_argument("The provided vector must have 3 elements");
    }

    auto matrix = HostView2D("cross_product_matrix", 3, 3);
    matrix(0, 0) = 0.;
    matrix(0, 1) = -vector(2);
    matrix(0, 2) = vector(1);
    matrix(1, 0) = vector(2);
    matrix(1, 1) = 0.;
    matrix(1, 2) = -vector(0);
    matrix(2, 0) = -vector(1);
    matrix(2, 1) = vector(0);
    matrix(2, 2) = 0.;

    return matrix;
}

HostView1D multiply_matrix_with_vector(HostView2D matrix, HostView1D vector) {
    if (matrix.extent(1) != vector.extent(0)) {
        throw std::invalid_argument(
            "The number of columns of the matrix must be equal to the number of rows of the vector"
        );
    }

    auto result = HostView1D("result", matrix.extent(0));
    auto entries = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, matrix.extent(0));
    auto multiply_row = [matrix, vector, result](int row) {
        double sum = 0.;
        for (size_t column = 0; column < matrix.extent(1); ++column) {
            sum += matrix(row, column) * vector(column);
        }
        result(row) = sum;
    };

    Kokkos::parallel_for(entries, multiply_row);

    return result;
}

HostView2D multiply_matrix_with_matrix(HostView2D matrix_a, HostView2D matrix_b) {
    auto n_rows = matrix_a.extent(0);
    auto n_columns = matrix_b.extent(1);

    if (n_rows != n_columns) {
        throw std::invalid_argument(
            "The number of columns of the first matrix must be equal to the number of rows of the "
            "second matrix"
        );
    }

    auto result = HostView2D("result", n_rows, n_columns);
    auto entries = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
        {0, 0}, {n_rows, n_columns}
    );
    auto multiply_row_column = [matrix_a, matrix_b, result](int row, int column) {
        double sum = 0.;
        for (size_t i = 0; i < matrix_a.extent(1); ++i) {
            sum += matrix_a(row, i) * matrix_b(i, column);
        }
        result(row, column) = sum;
    };

    Kokkos::parallel_for(entries, multiply_row_column);

    return result;
}

}  // namespace openturbine::rigid_pendulum
