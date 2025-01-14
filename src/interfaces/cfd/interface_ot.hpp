#pragma once

#include "src/interfaces/cfd/interface.hpp"
#include "src/model/model.hpp"

namespace openturbine::cfd {

class InterfaceOT : public Interface {
public:
    explicit InterfaceOT(const InterfaceInput& input);

    /// @brief  OpenTurbine class used for model construction
    Model model;

    /// @brief Turbine model input/output data
    Turbine turbine;

    /// @brief  OpenTurbine class for storing system state
    State state;

    /// @brief  OpenTurbine class for model elements (beams, masses, springs)
    Elements elements;

    /// @brief  OpenTurbine class for constraints tying elements together
    Constraints constraints;

    /// @brief  OpenTurbine class containing solution parameters
    StepParameters parameters;

    /// @brief  OpenTurbine class for solving the dynamic system
    Solver solver;

    /// @brief  OpenTurbine class state class for temporarily saving state
    State state_save;

    /// @brief Step forward in time
    void Step() override;

    /// @brief Save state for correction step
    void SaveState() override;

    /// @brief Restore state for correction step
    void RestoreState() override;
};

inline std::vector<double> kokkos_view_1D_to_vector(const Kokkos::View<double*>& view) {
    auto view_host = Kokkos::create_mirror(view);
    Kokkos::deep_copy(view_host, view);
    std::vector<double> values;
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        values.emplace_back(view_host(i));
    }
    return values;
}

inline std::vector<std::vector<double>> kokkos_view_2D_to_vector(const Kokkos::View<double**>& view
) {
    const Kokkos::View<double**> view_contiguous("view_contiguous", view.extent(0), view.extent(1));
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror(view_contiguous);
    Kokkos::deep_copy(view_host, view_contiguous);
    std::vector<std::vector<double>> values(view.extent(0));
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        for (size_t j = 0; j < view_host.extent(1); ++j) {
            values[i].emplace_back(view_host(i, j));
        }
    }
    return values;
}

}  // namespace openturbine::cfd
