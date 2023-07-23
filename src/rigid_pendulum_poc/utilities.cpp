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

}  // namespace openturbine::rigid_pendulum
