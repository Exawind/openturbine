#pragma once

#include <vtkCellType.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkLagrangeCurve.h>
#include <vtkLine.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include "elements/beams/beams.hpp"
#include "elements/beams/calculate_QP_deformation.hpp"
#include "elements/beams/interpolate_to_quadrature_points.hpp"
#include "math/quaternion_operations.hpp"
#include "state/state.hpp"
#include "system/beams/update_node_state.hpp"
#include "test_utilities.hpp"

namespace openturbine::tests {

template <typename DeviceType>
inline void WriteVTKBeamsQP(
    State<DeviceType>& state, Beams<DeviceType>& beams, const std::string& filename
) {
    // Compute state values at quadrature points
    auto range_policy = Kokkos::TeamPolicy<typename DeviceType::execution_space>(
        static_cast<int>(beams.num_elems), Kokkos::AUTO()
    );
    const auto smem = 3 * Kokkos::View<double* [7]>::shmem_size(beams.max_elem_nodes);
    range_policy.set_scratch_size(1, Kokkos::PerTeam(smem));

    Kokkos::parallel_for(
        "UpdateNodeState", range_policy,
        beams::UpdateNodeStatei<DeviceType>{
            state.q, state.v, state.vd, beams.node_state_indices, beams.num_nodes_per_element,
            beams.node_u, beams.node_u_dot, beams.node_u_ddot
        }
    );

    Kokkos::parallel_for(
        "InterpolateToQuadraturePoints", range_policy,
        InterpolateToQuadraturePoints<DeviceType>{
            beams.num_nodes_per_element, beams.num_qps_per_element, beams.shape_interp,
            beams.shape_deriv, beams.qp_jacobian, beams.node_u, beams.node_u_dot, beams.node_u_ddot,
            beams.qp_x0, beams.qp_r0, beams.qp_u, beams.qp_u_prime, beams.qp_r, beams.qp_r_prime,
            beams.qp_u_dot, beams.qp_omega, beams.qp_u_ddot, beams.qp_omega_dot, beams.qp_x
        }
    );

    // Get a copy of the beam element indices in the host space
    auto num_qps_per_element =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.num_qps_per_element);

    // Create a grid object and add the points to it.
    const auto gd = vtkNew<vtkUnstructuredGrid>();
    gd->Allocate(static_cast<vtkIdType>(beams.num_elems));
    auto first_qp = size_t{0U};
    for (size_t i = 0; i < beams.num_elems; ++i) {
        const auto num_qp = num_qps_per_element(i);
        const auto last_qp = first_qp + num_qp - 1U;
        std::vector<vtkIdType> pts;
        pts.push_back(static_cast<vtkIdType>(first_qp));  // first qp
        pts.push_back(static_cast<vtkIdType>(last_qp));   // last qp
        for (size_t j = first_qp; j < last_qp - 1U; ++j) {
            pts.push_back(static_cast<vtkIdType>(j));
        }
        gd->InsertNextCell(VTK_LAGRANGE_CURVE, static_cast<vtkIdType>(num_qp), pts.data());
        first_qp += num_qp;
    }

    //--------------------------------------------------------------------------
    // Position
    //--------------------------------------------------------------------------

    // Create qp position points
    auto qp_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.qp_x);
    const auto qp_pos = vtkNew<vtkPoints>();
    for (size_t j = 0; j < beams.num_elems; ++j) {
        for (size_t k = 0; k < num_qps_per_element(j); ++k) {
            qp_pos->InsertNextPoint(qp_x(j, k, 0), qp_x(j, k, 1), qp_x(j, k, 2));
        }
    }
    gd->SetPoints(qp_pos);

    //--------------------------------------------------------------------------
    // Orientation
    //--------------------------------------------------------------------------

    // Add orientation point data
    const auto orientation_x = vtkNew<vtkFloatArray>();
    orientation_x->SetNumberOfComponents(3);
    orientation_x->SetName("OrientationX");
    const auto orientation_y = vtkNew<vtkFloatArray>();
    orientation_y->SetNumberOfComponents(3);
    orientation_y->SetName("OrientationY");
    const auto orientation_z = vtkNew<vtkFloatArray>();
    orientation_z->SetNumberOfComponents(3);
    orientation_z->SetName("OrientationZ");
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto i = 0U; i < num_qps_per_element(el); ++i) {
            auto R = QuaternionToRotationMatrix(
                {qp_x(el, i, 3), qp_x(el, i, 4), qp_x(el, i, 5), qp_x(el, i, 6)}
            );
            const auto ori_x = std::array{R[0][0], R[1][0], R[2][0]};
            const auto ori_y = std::array{R[0][1], R[1][1], R[2][1]};
            const auto ori_z = std::array{R[0][2], R[1][2], R[2][2]};
            orientation_x->InsertNextTuple(ori_x.data());
            orientation_y->InsertNextTuple(ori_y.data());
            orientation_z->InsertNextTuple(ori_z.data());
        }
    }
    gd->GetPointData()->AddArray(orientation_x);
    gd->GetPointData()->AddArray(orientation_y);
    gd->GetPointData()->AddArray(orientation_z);

    //--------------------------------------------------------------------------
    // Velocity
    //--------------------------------------------------------------------------

    // Add translational velocity point data
    const auto translational_velocity = vtkNew<vtkFloatArray>();
    translational_velocity->SetNumberOfComponents(3);
    translational_velocity->SetName("TranslationalVelocity");
    const auto qp_u_dot = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.qp_u_dot);
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_qps_per_element(el); ++p) {
            const auto tv = std::array{qp_u_dot(el, p, 0), qp_u_dot(el, p, 1), qp_u_dot(el, p, 2)};
            translational_velocity->InsertNextTuple(tv.data());
        }
    }
    gd->GetPointData()->AddArray(translational_velocity);

    // Add rotational velocity point data
    const auto rotational_velocity = vtkNew<vtkFloatArray>();
    rotational_velocity->SetNumberOfComponents(3);
    rotational_velocity->SetName("RotationalVelocity");
    const auto qp_omega = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.qp_omega);
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_qps_per_element(el); ++p) {
            const auto rv = std::array{qp_omega(el, p, 0), qp_omega(el, p, 1), qp_omega(el, p, 2)};
            rotational_velocity->InsertNextTuple(rv.data());
        }
    }
    gd->GetPointData()->AddArray(rotational_velocity);

    //--------------------------------------------------------------------------
    // Acceleration
    //--------------------------------------------------------------------------

    // // Add translational acceleration point data
    // vtkNew<vtkFloatArray> translational_accel;
    // translational_accel->SetNumberOfComponents(3);
    // translational_accel->SetName("TranslationalAcceleration");
    // const auto qp_u_ddot = Kokkos::create_mirror(beams.qp_u_ddot);
    // Kokkos::deep_copy(qp_u_ddot, beams.qp_u_ddot);
    // for (auto el = 0U; el < beams.num_elems; ++el) {
    //     for (auto p = 0U; p < num_qps_per_element(el); ++p) {
    //         const auto tv =
    //             std::array{qp_u_ddot(el, p, 0), qp_u_ddot(el, p, 1), qp_u_ddot(el, p, 2)};
    //         translational_accel->InsertNextTuple(tv.data());
    //     }
    // }
    // gd->GetPointData()->AddArray(translational_accel);

    // // Add rotational acceleration point data
    // vtkNew<vtkFloatArray> rotational_accel;
    // rotational_accel->SetNumberOfComponents(3);
    // rotational_accel->SetName("RotationalAcceleration");
    // const auto qp_omega_dot = Kokkos::create_mirror(beams.qp_omega_dot);
    // Kokkos::deep_copy(qp_omega_dot, beams.qp_omega_dot);
    // for (auto el = 0U; el < beams.num_elems; ++el) {
    //     for (auto p = 0U; p < num_qps_per_element(el); ++p) {
    //         const auto rv =
    //             std::array{qp_omega_dot(el, p, 0), qp_omega_dot(el, p, 1), qp_omega_dot(el, p,
    //             2)};
    //         rotational_accel->InsertNextTuple(rv.data());
    //     }
    // }
    // gd->GetPointData()->AddArray(rotational_accel);

    //--------------------------------------------------------------------------
    // Loads
    //--------------------------------------------------------------------------

    const auto qp_Fe = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.qp_Fe);

    // Add translational acceleration point data
    const auto force = vtkNew<vtkFloatArray>();
    force->SetNumberOfComponents(3);
    force->SetName("Force");

    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_qps_per_element(el); ++p) {
            const auto tv = std::array{qp_Fe(el, p, 0), qp_Fe(el, p, 1), qp_Fe(el, p, 2)};
            force->InsertNextTuple(tv.data());
        }
    }
    gd->GetPointData()->AddArray(force);

    // Add rotational acceleration point data
    const auto moment = vtkNew<vtkFloatArray>();
    moment->SetNumberOfComponents(3);
    moment->SetName("Moment");
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_qps_per_element(el); ++p) {
            const auto rv = std::array{qp_Fe(el, p, 3), qp_Fe(el, p, 4), qp_Fe(el, p, 5)};
            moment->InsertNextTuple(rv.data());
        }
    }
    gd->GetPointData()->AddArray(moment);

    //--------------------------------------------------------------------------
    // Deformation
    //--------------------------------------------------------------------------

    // Add deformation point data
    const auto deformation_vector = vtkNew<vtkFloatArray>();
    deformation_vector->SetNumberOfComponents(3);
    deformation_vector->SetName("DeformationVector");
    // TODO: use hierarchical parallelism to make this function better.
    Kokkos::parallel_for(
        "CalculateQPDeformation",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {beams.num_elems, beams.max_elem_qps}),
        CalculateQPDeformation{
            beams.num_qps_per_element,
            beams.qp_x0,
            beams.qp_r,
            beams.qp_x,
            beams.qp_deformation,
        }
    );
    auto qp_deformation =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.qp_deformation);
    for (auto i = 0U; i < beams.num_elems; ++i) {
        // Get the undeformed shape
        auto num_qps = num_qps_per_element(i);
        for (auto j = 0U; j < num_qps; ++j) {
            const Array_3 u{
                qp_deformation(i, j, 0),
                qp_deformation(i, j, 1),
                qp_deformation(i, j, 2),
            };
            deformation_vector->InsertNextTuple(u.data());
        }
    }
    gd->GetPointData()->AddArray(deformation_vector);

    // Write the file
    const auto writer = vtkNew<vtkXMLUnstructuredGridWriter>();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(gd);
    writer->SetDataModeToAscii();
    writer->Write();
}

template <typename DeviceType>
inline void WriteVTKBeamsNodes(
    State<DeviceType>& state, Beams<DeviceType>& beams, const std::string& filename
) {
    // Compute value of state at beam nodes
    auto range_policy = Kokkos::TeamPolicy<typename DeviceType::execution_space>(
        static_cast<int>(beams.num_elems), Kokkos::AUTO()
    );
    const auto smem = 3 * Kokkos::View<double* [7]>::shmem_size(beams.max_elem_nodes);
    range_policy.set_scratch_size(1, Kokkos::PerTeam(smem));

    Kokkos::parallel_for(
        "UpdateNodeState", range_policy,
        beams::UpdateNodeState<DeviceType>{
            state.q, state.v, state.vd, beams.node_state_indices, beams.num_nodes_per_element,
            beams.node_u, beams.node_u_dot, beams.node_u_ddot
        }
    );

    // Get a copy of the beam element indices in the host space
    auto num_nodes_per_element =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.num_nodes_per_element);

    // Create a grid object and add the points to it.
    const auto gd = vtkNew<vtkUnstructuredGrid>();
    gd->Allocate(static_cast<vtkIdType>(beams.num_elems));
    size_t first_node{0U};
    for (auto i = 0U; i < beams.num_elems; ++i) {
        const auto num_node = num_nodes_per_element(i);
        const auto last_node = first_node + num_node - 1U;
        std::vector<vtkIdType> pts;
        pts.push_back(static_cast<vtkIdType>(first_node));  // first qp
        pts.push_back(static_cast<vtkIdType>(last_node));   // last qp
        for (auto j = first_node; j < last_node - 1U; ++j) {
            pts.push_back(static_cast<vtkIdType>(j));
        }
        gd->InsertNextCell(VTK_LAGRANGE_CURVE, static_cast<vtkIdType>(num_node), pts.data());
        first_node += num_node;
    }

    //--------------------------------------------------------------------------
    // Position
    //--------------------------------------------------------------------------

    auto node_x0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.node_x0);

    auto node_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.node_u);

    // Create qp position points
    const auto node_pos = vtkNew<vtkPoints>();
    for (size_t j = 0; j < beams.num_elems; ++j) {
        for (size_t k = 0; k < num_nodes_per_element(j); ++k) {
            node_pos->InsertNextPoint(
                node_x0(j, k, 0) + node_u(j, k, 0), node_x0(j, k, 1) + node_u(j, k, 1),
                node_x0(j, k, 2) + node_u(j, k, 2)
            );
        }
    }
    gd->SetPoints(node_pos);

    // Add orientation point data
    const auto orientation_x = vtkNew<vtkFloatArray>();
    orientation_x->SetNumberOfComponents(3);
    orientation_x->SetName("OrientationX");
    const auto orientation_y = vtkNew<vtkFloatArray>();
    orientation_y->SetNumberOfComponents(3);
    orientation_y->SetName("OrientationY");
    const auto orientation_z = vtkNew<vtkFloatArray>();
    orientation_z->SetNumberOfComponents(3);
    orientation_z->SetName("OrientationZ");
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto i = 0U; i < num_nodes_per_element(el); ++i) {
            auto q = QuaternionCompose(
                {node_u(el, i, 3), node_u(el, i, 4), node_u(el, i, 5), node_u(el, i, 6)},
                {node_x0(el, i, 3), node_x0(el, i, 4), node_x0(el, i, 5), node_x0(el, i, 6)}
            );
            auto R = QuaternionToRotationMatrix(q);
            const auto ori_x = std::array{R[0][0], R[1][0], R[2][0]};
            const auto ori_y = std::array{R[0][1], R[1][1], R[2][1]};
            const auto ori_z = std::array{R[0][2], R[1][2], R[2][2]};
            orientation_x->InsertNextTuple(ori_x.data());
            orientation_y->InsertNextTuple(ori_y.data());
            orientation_z->InsertNextTuple(ori_z.data());
        }
    }
    gd->GetPointData()->AddArray(orientation_x);
    gd->GetPointData()->AddArray(orientation_y);
    gd->GetPointData()->AddArray(orientation_z);

    //--------------------------------------------------------------------------
    // Velocity
    //--------------------------------------------------------------------------

    const auto node_u_dot =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.node_u_dot);

    // Add translational velocity point data
    const auto translational_velocity = vtkNew<vtkFloatArray>();
    translational_velocity->SetNumberOfComponents(3);
    translational_velocity->SetName("TranslationalVelocity");
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_nodes_per_element(el); ++p) {
            const auto tv =
                std::array{node_u_dot(el, p, 0), node_u_dot(el, p, 1), node_u_dot(el, p, 2)};
            translational_velocity->InsertNextTuple(tv.data());
        }
    }
    gd->GetPointData()->AddArray(translational_velocity);

    // Add rotational velocity point data
    const auto rotational_velocity = vtkNew<vtkFloatArray>();
    rotational_velocity->SetNumberOfComponents(3);
    rotational_velocity->SetName("RotationalVelocity");
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_nodes_per_element(el); ++p) {
            const auto rv =
                std::array{node_u_dot(el, p, 3), node_u_dot(el, p, 4), node_u_dot(el, p, 5)};
            rotational_velocity->InsertNextTuple(rv.data());
        }
    }
    gd->GetPointData()->AddArray(rotational_velocity);

    //--------------------------------------------------------------------------
    // Acceleration
    //--------------------------------------------------------------------------

    // const auto node_u_ddot = Kokkos::create_mirror(beams.node_u_ddot);
    // Kokkos::deep_copy(node_u_ddot, beams.node_u_ddot);

    // // Add translational acceleration point data
    // vtkNew<vtkFloatArray> translational_accel;
    // translational_accel->SetNumberOfComponents(3);
    // translational_accel->SetName("TranslationalAcceleration");

    // for (auto el = 0U; el < beams.num_elems; ++el) {
    //     for (auto p = 0U; p < num_nodes_per_element(el); ++p) {
    //         const auto tv =
    //             std::array{node_u_ddot(el, p, 0), node_u_ddot(el, p, 1), node_u_ddot(el, p, 2)};
    //         translational_accel->InsertNextTuple(tv.data());
    //     }
    // }
    // gd->GetPointData()->AddArray(translational_accel);

    // // Add rotational acceleration point data
    // vtkNew<vtkFloatArray> rotational_accel;
    // rotational_accel->SetNumberOfComponents(3);
    // rotational_accel->SetName("RotationalAcceleration");
    // for (auto el = 0U; el < beams.num_elems; ++el) {
    //     for (auto p = 0U; p < num_nodes_per_element(el); ++p) {
    //         const auto rv =
    //             std::array{node_u_ddot(el, p, 3), node_u_ddot(el, p, 4), node_u_ddot(el, p, 5)};
    //         rotational_accel->InsertNextTuple(rv.data());
    //     }
    // }
    // gd->GetPointData()->AddArray(rotational_accel);

    //--------------------------------------------------------------------------
    // Loads
    //--------------------------------------------------------------------------

    const auto node_FX = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), beams.node_FX);

    // Add translational acceleration point data
    const auto force = vtkNew<vtkFloatArray>();
    force->SetNumberOfComponents(3);
    force->SetName("Force");

    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_nodes_per_element(el); ++p) {
            const auto tv = std::array{node_FX(el, p, 0), node_FX(el, p, 1), node_FX(el, p, 2)};
            force->InsertNextTuple(tv.data());
        }
    }
    gd->GetPointData()->AddArray(force);

    // Add rotational acceleration point data
    const auto moment = vtkNew<vtkFloatArray>();
    moment->SetNumberOfComponents(3);
    moment->SetName("Moment");
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_nodes_per_element(el); ++p) {
            const auto rv = std::array{node_FX(el, p, 3), node_FX(el, p, 4), node_FX(el, p, 5)};
            moment->InsertNextTuple(rv.data());
        }
    }
    gd->GetPointData()->AddArray(moment);

    //--------------------------------------------------------------------------
    // Write file
    //--------------------------------------------------------------------------

    const auto writer = vtkNew<vtkXMLUnstructuredGridWriter>();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(gd);
    writer->SetDataModeToAscii();
    writer->Write();
}

inline void WriteVTKPoint(
    Array_7 position, Array_6 velocity, Array_6 acceleration, const std::string& filename
) {
    // Create a grid object and add the points to it.
    const auto gd = vtkNew<vtkUnstructuredGrid>();
    gd->Allocate(1);
    std::vector<vtkIdType> pts;
    pts.push_back(static_cast<vtkIdType>(1));  // first qp
    gd->InsertNextCell(VTK_VERTEX, static_cast<vtkIdType>(1), pts.data());

    Array_3 tmp{0., 0., 0.};

    //--------------------------------------------------------------------------
    // Position
    //--------------------------------------------------------------------------

    // Create qp position points
    const auto point_pos = vtkNew<vtkPoints>();
    point_pos->InsertNextPoint(position[0], position[1], position[2]);
    gd->SetPoints(point_pos);

    // Add orientation point data
    const auto orientation_x = vtkNew<vtkFloatArray>();
    orientation_x->SetNumberOfComponents(3);
    orientation_x->SetName("OrientationX");
    const auto orientation_y = vtkNew<vtkFloatArray>();
    orientation_y->SetNumberOfComponents(3);
    orientation_y->SetName("OrientationY");
    const auto orientation_z = vtkNew<vtkFloatArray>();
    orientation_z->SetNumberOfComponents(3);
    orientation_z->SetName("OrientationZ");
    auto R = QuaternionToRotationMatrix(Array_4{position[3], position[4], position[5], position[6]});
    const auto ori_x = std::array{R[0][0], R[1][0], R[2][0]};
    const auto ori_y = std::array{R[0][1], R[1][1], R[2][1]};
    const auto ori_z = std::array{R[0][2], R[1][2], R[2][2]};
    orientation_x->InsertNextTuple(ori_x.data());
    orientation_y->InsertNextTuple(ori_y.data());
    orientation_z->InsertNextTuple(ori_z.data());
    gd->GetPointData()->AddArray(orientation_x);
    gd->GetPointData()->AddArray(orientation_y);
    gd->GetPointData()->AddArray(orientation_z);

    //--------------------------------------------------------------------------
    // Velocity
    //--------------------------------------------------------------------------

    // Add translational velocity point data
    const auto translational_velocity = vtkNew<vtkFloatArray>();
    translational_velocity->SetNumberOfComponents(3);
    translational_velocity->SetName("TranslationalVelocity");
    tmp = Array_3{velocity[0], velocity[1], velocity[2]};
    translational_velocity->InsertNextTuple(tmp.data());
    gd->GetPointData()->AddArray(translational_velocity);

    // Add rotational velocity point data
    const auto rotational_velocity = vtkNew<vtkFloatArray>();
    rotational_velocity->SetNumberOfComponents(3);
    rotational_velocity->SetName("RotationalVelocity");
    tmp = Array_3{velocity[3], velocity[4], velocity[5]};
    rotational_velocity->InsertNextTuple(tmp.data());
    gd->GetPointData()->AddArray(rotational_velocity);

    //--------------------------------------------------------------------------
    // Acceleration
    //--------------------------------------------------------------------------

    // Add translational acceleration point data
    const auto translational_accel = vtkNew<vtkFloatArray>();
    translational_accel->SetNumberOfComponents(3);
    translational_accel->SetName("TranslationalAcceleration");
    tmp = Array_3{acceleration[0], acceleration[1], acceleration[2]};
    translational_accel->InsertNextTuple(tmp.data());
    gd->GetPointData()->AddArray(translational_accel);

    // Add rotational acceleration point data
    const auto rotational_accel = vtkNew<vtkFloatArray>();
    rotational_accel->SetNumberOfComponents(3);
    rotational_accel->SetName("RotationalAcceleration");
    tmp = Array_3{velocity[3], velocity[4], velocity[5]};
    rotational_accel->InsertNextTuple(tmp.data());
    gd->GetPointData()->AddArray(rotational_accel);

    //--------------------------------------------------------------------------
    // Write file
    //--------------------------------------------------------------------------

    const auto writer = vtkNew<vtkXMLUnstructuredGridWriter>();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(gd);
    writer->SetDataModeToAscii();
    writer->Write();
}

}  // namespace openturbine::tests
