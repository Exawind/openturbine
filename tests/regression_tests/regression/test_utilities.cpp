#include "test_utilities.hpp"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace openturbine::tests {

std::filesystem::path FindProjectRoot() {
    std::filesystem::path currentPath = std::filesystem::current_path();

    while (!currentPath.empty()) {
        if (std::filesystem::exists(currentPath / "CMakeLists.txt")) {
            return currentPath;
        }
        currentPath = currentPath.parent_path();
    }

    throw std::runtime_error("Could not find project root directory. CMakeLists.txt not found.");
}

void RemoveDirectoryWithRetries(const std::filesystem::path& dir, int retries, int delayMs) {
    for (auto i = 0; i < retries; ++i) {
        try {
            std::filesystem::remove_all(dir);
            return;
        } catch (const std::filesystem::filesystem_error& e) {
            if (i < retries - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
            } else {
                std::cerr << "Failed to remove directory: " << dir << "\n";
                throw;
            }
        }
    }
}

Kokkos::View<double**> create_diagonal_matrix(const std::vector<double>& values) {
    auto matrix = Kokkos::View<double**>("matrix", values.size(), values.size());
    auto matrix_host = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, matrix);

    for (auto index = 0U; index < values.size(); ++index) {
        matrix_host(index, index) = values[index];
    }
    Kokkos::deep_copy(matrix, matrix_host);

    return matrix;
}

void expect_kokkos_view_1D_equal(
    const Kokkos::View<const double*>& view, const std::vector<double>& expected, double epsilon
) {
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    for (auto i = 0U; i < view_host.extent(0); ++i) {
        EXPECT_NEAR(view_host(i), expected[i], epsilon);
    }
}

void expect_kokkos_view_2D_equal(
    const Kokkos::View<const double**, Kokkos::LayoutStride>& view,
    const std::vector<std::vector<double>>& expected, double epsilon
) {
    const Kokkos::View<double**> view_contiguous("view_contiguous", view.extent(0), view.extent(1));
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_contiguous);
    for (auto i = 0U; i < view_host.extent(0); ++i) {
        for (auto j = 0U; j < view_host.extent(1); ++j) {
            EXPECT_NEAR(view_host(i, j), expected[i][j], epsilon);
        }
    }
}

std::vector<double> kokkos_view_1D_to_vector(const Kokkos::View<double*>& view) {
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    std::vector<double> values;
    for (auto i = 0U; i < view_host.extent(0); ++i) {
        values.emplace_back(view_host(i));
    }
    return values;
}

std::vector<std::vector<double>> kokkos_view_2D_to_vector(const Kokkos::View<double**>& view) {
    const Kokkos::View<double**> view_contiguous("view_contiguous", view.extent(0), view.extent(1));
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_contiguous);
    std::vector<std::vector<double>> values(view.extent(0));
    for (auto i = 0U; i < view_host.extent(0); ++i) {
        for (auto j = 0U; j < view_host.extent(1); ++j) {
            values[i].emplace_back(view_host(i, j));
        }
    }
    return values;
}

std::vector<std::vector<std::vector<double>>> kokkos_view_3D_to_vector(
    const Kokkos::View<double***>& view
) {
    const Kokkos::View<double***> view_contiguous(
        "view_contiguous", view.extent(0), view.extent(1), view.extent(2)
    );
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_contiguous);
    std::vector<std::vector<std::vector<double>>> values(view.extent(0));
    for (auto i = 0U; i < view_host.extent(0); ++i) {
        for (auto j = 0U; j < view_host.extent(1); ++j) {
            for (auto k = 0U; k < view_host.extent(2); ++k) {
                values[i][j].emplace_back(view_host(i, j, k));
            }
        }
    }
    return values;
}

void expect_kokkos_view_3D_equal(
    const Kokkos::View<const double***, Kokkos::LayoutStride>& view,
    const std::vector<std::vector<std::vector<double>>>& expected, double epsilon
) {
    const Kokkos::View<double***> view_contiguous(
        "view_contiguous", view.extent(0), view.extent(1), view.extent(2)
    );
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_contiguous);
    for (auto i = 0U; i < view_host.extent(0); ++i) {
        for (auto j = 0U; j < view_host.extent(1); ++j) {
            for (auto k = 0U; k < view_host.extent(2); ++k) {
                EXPECT_NEAR(view_host(i, j, k), expected[i][j][k], epsilon);
            }
        }
    }
}

}  // namespace openturbine::tests
