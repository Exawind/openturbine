#pragma once

#include "host_state.hpp"

namespace openturbine::interfaces {

struct NodeData {
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
    void ClearLoads() { this->loads = {0., 0., 0., 0., 0., 0.}; }

    /// @brief Populates node position, displacement, velocity, acceleration from state data
    /// @param node
    template <typename DeviceType>
    void UpdateMotion(const HostState<DeviceType>& host_state) {
        for (auto i = 0U; i < kLieGroupComponents; ++i) {
            this->position[i] = host_state.x(this->id, i);
            this->displacement[i] = host_state.q(this->id, i);
        }
        for (auto i = 0U; i < kLieAlgebraComponents; ++i) {
            this->velocity[i] = host_state.v(this->id, i);
            this->acceleration[i] = host_state.vd(this->id, i);
        }
    }
};

}  // namespace openturbine::interfaces
