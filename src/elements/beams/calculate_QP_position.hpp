#pragma once

#include <Kokkos_Core.hpp>

#include "math/quaternion_operations.hpp"

namespace openturbine {

/**
 * @brief Functor to calculate current position and orientation at quadrature points
 *
 * This functor updates the current position and orientation (qp_x_) of quadrature points
 * by combining:
 * - Initial position (qp_x0_) with displacement (qp_u_) for translational components
 * - Initial orientation (qp_r0_) with rotation (qp_r_) for rotational components using quaternions
 *
 * @param i_elem Element index
 * @param qp_x0_ Initial quadrature point positions (num_elems x num_qps x 3)
 * @param qp_u_ Displacements at quadrature points (num_elems x num_qps x 3)
 * @param qp_r0_ Initial orientations as quaternions (num_elems x num_qps x 4)
 * @param qp_r_ Rotations as quaternions (num_elems x num_qps x 4)
 * @param qp_x_ Output current positions and orientations (num_elems x num_qps x 7)
 *              where [0:3] = position, [3:7] = orientation quaternion
 */
template <typename DeviceType>
struct CalculateQPPosition {
    size_t i_elem;
    typename Kokkos::View<double** [3], DeviceType>::const_type qp_x0_;
    typename Kokkos::View<double** [3], DeviceType>::const_type qp_u_;
    typename Kokkos::View<double** [4], DeviceType>::const_type qp_r0_;
    typename Kokkos::View<double** [4], DeviceType>::const_type qp_r_;
    Kokkos::View<double** [7], DeviceType> qp_x_;

    KOKKOS_FUNCTION void operator()(const int i_qp) const {
        // Calculate current position
        qp_x_(i_elem, i_qp, 0) = qp_x0_(i_elem, i_qp, 0) + qp_u_(i_elem, i_qp, 0);
        qp_x_(i_elem, i_qp, 1) = qp_x0_(i_elem, i_qp, 1) + qp_u_(i_elem, i_qp, 1);
        qp_x_(i_elem, i_qp, 2) = qp_x0_(i_elem, i_qp, 2) + qp_u_(i_elem, i_qp, 2);

        // Calculate current orientation
        auto RR0_data = Kokkos::Array<double, 4>{};
        auto RR0 = Kokkos::View<double[4], DeviceType>(RR0_data.data());
        QuaternionCompose(
            Kokkos::subview(qp_r_, i_elem, i_qp, Kokkos::ALL),
            Kokkos::subview(qp_r0_, i_elem, i_qp, Kokkos::ALL), RR0
        );
        qp_x_(i_elem, i_qp, 3) = RR0(0);
        qp_x_(i_elem, i_qp, 4) = RR0(1);
        qp_x_(i_elem, i_qp, 5) = RR0(2);
        qp_x_(i_elem, i_qp, 6) = RR0(3);
    }
};

}  // namespace openturbine
