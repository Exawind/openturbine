#pragma once

#include "state/state.hpp"

namespace openturbine::interfaces {

/**
 * @brief Host-side mirror of the simulation state for a given time increment
 *
 * @details This struct maintains host-local copies of the key state variables required
 * for the dynamic simulation, including:
 * - Position: Current position and orientation [x, y, z, qw, qx, qy, qz] -> 7 x 1
 * - Displacement: Current displacement and rotation [δx, δy, δz, δqw, δqx, δqy, δqz] -> 7 x 1
 * - Velocity: Current linear and angular velocities [vx, vy, vz, wx, wy, wz] -> 6 x 1
 * - Acceleration: Current linear and angular accelerations [ax, ay, az, αx, αy, αz] -> 6 x 1
 *
 * @note This struct serves as a host-side mirror of the device state, allowing for
 * efficient data transfer between device and host memory. It's primarily used for
 * the interfaces, I/O operations etc.
 */
template <typename DeviceType>
struct HostState {
    /// @brief Host local copy of current position
    typename Kokkos::View<double* [7], DeviceType>::HostMirror x;

    /// @brief Host local copy of current displacement
    typename Kokkos::View<double* [7], DeviceType>::HostMirror q;

    /// @brief Host local copy of current velocity
    typename Kokkos::View<double* [6], DeviceType>::HostMirror v;

    /// @brief Host local copy of current acceleration
    typename Kokkos::View<double* [6], DeviceType>::HostMirror vd;

    /// @brief Host local copy of external forces
    typename Kokkos::View<double* [6], DeviceType>::HostMirror f;

    /// @brief  Construct host state from state
    explicit HostState(const State<DeviceType>& state)
        : x(Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.x)),
          q(Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.q)),
          v(Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.v)),
          vd(Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.vd)),
          f(Kokkos::create_mirror_view(Kokkos::WithoutInitializing, state.f)) {}

    /// @brief Copy state data to host state
    void CopyFromState(const State<DeviceType>& state) const {
        Kokkos::deep_copy(this->x, state.x);
        Kokkos::deep_copy(this->q, state.q);
        Kokkos::deep_copy(this->v, state.v);
        Kokkos::deep_copy(this->vd, state.vd);
    }
};

}  // namespace openturbine::interfaces
