#pragma once

#include "interfaces/node_data.hpp"
#include "state/state.hpp"
#include "types.hpp"

namespace openturbine::interfaces {

/* @brief Container for storing the complete system state of the simulation at a given
 * time increment
 * @details This struct maintains all state variables required for the dynamic simulation,
 * including:
 * - Node configuration and freedom allocation
 * - Position and rotation states (current, previous, and initial)
 * - Velocity and acceleration states
 * - Tangent matrix
 */
struct HostState {
    /// @brief Host local copy of current position
    Kokkos::View<double* [7]>::HostMirror x;

    /// @brief Host local copy of current displacement
    Kokkos::View<double* [7]>::HostMirror q;

    /// @brief Host local copy of current velocity
    Kokkos::View<double* [6]>::HostMirror v;

    /// @brief Host local copy of current acceleration
    Kokkos::View<double* [6]>::HostMirror vd;

    /// @brief  Construct host state from state
    /// @param state
    HostState(const State& state)
        : x("host_state_x", state.num_system_nodes),
          q("host_state_q", state.num_system_nodes),
          v("host_state_v", state.num_system_nodes),
          vd("host_state_vd", state.num_system_nodes) {}

    /// @brief Copy state data to host state
    /// @param state
    void CopyFromState(const State& state) {
        Kokkos::deep_copy(this->x, state.x);
        Kokkos::deep_copy(this->q, state.q);
        Kokkos::deep_copy(this->v, state.v);
        Kokkos::deep_copy(this->vd, state.vd);
    }

    /// @brief Populates node position, displacement, velocity, acceleration from state data
    /// @param node
    void SetNodeMotion(NodeData& node) {
        for (auto i = 0U; i < kLieGroupComponents; ++i) {
            node.position[i] = this->x(node.id, i);
            node.displacement[i] = this->q(node.id, i);
        }
        for (auto i = 0U; i < kLieAlgebraComponents; ++i) {
            node.velocity[i] = this->v(node.id, i);
            node.acceleration[i] = this->vd(node.id, i);
        }
    }
};

}  // namespace openturbine::interfaces
