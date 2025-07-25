#pragma once

#include <array>

#include <Kokkos_Core.hpp>

namespace openturbine::interfaces {

template <typename DeviceType>
struct HostState;

struct NodeData {
    using DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

    /// @brief Node identifier in model
    size_t id;

    /// @brief Absolute position of node in global coordinates
    std::array<double, 7> position{0., 0., 0., 1., 0., 0., 0.};

    /// @brief Displacement from reference position
    std::array<double, 7> displacement{0., 0., 0., 1., 0., 0., 0.};

    /// @brief Velocity of node in global coordinates
    std::array<double, 6> velocity{0., 0., 0., 0., 0., 0.};

    /// @brief Acceleration of node in global coordinates
    std::array<double, 6> acceleration{0., 0., 0., 0., 0., 0.};

    /// @brief Point loads/moment applied to node in global coordinates
    std::array<double, 6> loads{0., 0., 0., 0., 0., 0.};

    /// @brief Node data constructor
    /// @param id Node identifier in model
    explicit NodeData(size_t id_) : id(id_) {}

    /// @brief Set point loads and moments to zero
    void ClearLoads();

    /// @brief Populates node position, displacement, velocity, acceleration from state data
    /// @param node
    void GetMotion(const HostState<DeviceType>& host_state);

    /// @brief Updates the node loads in the host state
    /// @param host_state Host state to update
    void SetLoads(HostState<DeviceType>& host_state) const;
};

}  // namespace openturbine::interfaces
