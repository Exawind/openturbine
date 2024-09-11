#pragma once

#include <vtkCellArray.h>
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

#include "test_utilities.hpp"

#include "src/beams/beams.hpp"
#include "src/math/quaternion_operations.hpp"

namespace openturbine::tests {

inline void BeamsWriteVTK(Beams& beams, std::string filename) {
    // Get a copy of the beam element indices in the host space
    auto num_qps_per_element = Kokkos::create_mirror(beams.num_qps_per_element);
    Kokkos::deep_copy(num_qps_per_element, beams.num_qps_per_element);

    // Create a grid object and add the points to it.
    vtkNew<vtkUnstructuredGrid> gd;
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

    // Create qp position points
    auto qp_x0 = Kokkos::create_mirror(beams.qp_x0);
    Kokkos::deep_copy(qp_x0, beams.qp_x0);
    auto qp_x = Kokkos::create_mirror(beams.qp_u);
    Kokkos::deep_copy(qp_x, beams.qp_u);
    for (size_t j = 0; j < beams.num_elems; ++j) {
        for (size_t k = 0; k < num_qps_per_element(j); ++k) {
            for (size_t l = 0; l < qp_x.extent(2); ++l) {
                qp_x(j, k, l) += qp_x0(j, k, l);
            }
        }
    }
    vtkNew<vtkPoints> qp_pos;
    for (size_t j = 0; j < beams.num_elems; ++j) {
        for (size_t k = 0; k < num_qps_per_element(j); ++k) {
            qp_pos->InsertNextPoint(qp_x(j, k, 0), qp_x(j, k, 1), qp_x(j, k, 2));
        }
    }
    gd->SetPoints(qp_pos);

    // Add orientation point data
    vtkNew<vtkFloatArray> orientation_x;
    orientation_x->SetNumberOfComponents(3);
    orientation_x->SetName("OrientationX");
    vtkNew<vtkFloatArray> orientation_y;
    orientation_y->SetNumberOfComponents(3);
    orientation_y->SetName("OrientationY");
    vtkNew<vtkFloatArray> orientation_z;
    orientation_z->SetNumberOfComponents(3);
    orientation_z->SetName("OrientationZ");
    auto qp_r = Kokkos::create_mirror(beams.qp_r);
    Kokkos::deep_copy(qp_r, beams.qp_r);
    auto qp_r0 = Kokkos::create_mirror(beams.qp_r0);
    Kokkos::deep_copy(qp_r0, beams.qp_r0);
    // auto qp_r = kokkos_view_3D_to_vector(beams.qp_r);
    // auto qp_r0 = kokkos_view_3D_to_vector(beams.qp_r0);
    for (size_t el = 0; el < beams.num_elems; ++el) {
        for (size_t i = 0; i < num_qps_per_element(el); ++i) {
            Array_4 r{qp_r(el, i, 0), qp_r(el, i, 1), qp_r(el, i, 2), qp_r(el, i, 3)};
            Array_4 r0{qp_r0(el, i, 0), qp_r0(el, i, 1), qp_r0(el, i, 2), qp_r0(el, i, 3)};
            auto R = QuaternionToRotationMatrix(QuaternionCompose(r, r0));
            double ori_x[3] = {R[0][0], R[1][0], R[2][0]};
            double ori_y[3] = {R[0][1], R[1][1], R[2][1]};
            double ori_z[3] = {R[0][2], R[1][2], R[2][2]};
            orientation_x->InsertNextTuple(ori_x);
            orientation_y->InsertNextTuple(ori_y);
            orientation_z->InsertNextTuple(ori_z);
        }
    }
    gd->GetPointData()->AddArray(orientation_x);
    gd->GetPointData()->AddArray(orientation_y);
    gd->GetPointData()->AddArray(orientation_z);

    // Add translational velocity point data
    vtkNew<vtkFloatArray> translational_velocity;
    translational_velocity->SetNumberOfComponents(3);
    translational_velocity->SetName("TranslationalVelocity");
    const auto qp_u_dot = Kokkos::create_mirror(beams.qp_u_dot);
    Kokkos::deep_copy(qp_u_dot, beams.qp_u_dot);
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_qps_per_element(el); ++p) {
            const auto tv = std::array{qp_u_dot(el, p, 0), qp_u_dot(el, p, 1), qp_u_dot(el, p, 2)};
            translational_velocity->InsertNextTuple(tv.data());
        }
    }
    gd->GetPointData()->AddArray(translational_velocity);

    // Add rotational velocity point data
    vtkNew<vtkFloatArray> rotational_velocity;
    rotational_velocity->SetNumberOfComponents(3);
    rotational_velocity->SetName("RotationalVelocity");
    const auto qp_omega = Kokkos::create_mirror(beams.qp_omega);
    Kokkos::deep_copy(qp_omega, beams.qp_omega);
    for (auto el = 0U; el < beams.num_elems; ++el) {
        for (auto p = 0U; p < num_qps_per_element(el); ++p) {
            const auto rv = std::array{qp_omega(el, p, 0), qp_omega(el, p, 1), qp_omega(el, p, 2)};
            rotational_velocity->InsertNextTuple(rv.data());
        }
    }
    gd->GetPointData()->AddArray(rotational_velocity);

    // Add deformation point data
    vtkNew<vtkFloatArray> deformation_vector;
    deformation_vector->SetNumberOfComponents(3);
    deformation_vector->SetName("DeformationVector");
    for (size_t i = 0; i < beams.num_elems; ++i) {
        // Get the undeformed shape
        auto num_qps = num_qps_per_element(i);
        Array_3 x0_root{qp_x0(i, 0, 0), qp_x0(i, 0, 1), qp_x0(i, 0, 2)};
        Array_3 x_root{qp_x(i, 0, 0), qp_x(i, 0, 1), qp_x(i, 0, 2)};
        Array_4 r_u{qp_r(i, 0, 0), qp_r(i, 0, 1), qp_r(i, 0, 2), qp_r(i, 0, 3)};

        for (size_t j = 0; j < num_qps; ++j) {
            Array_3 x{qp_x(i, j, 0), qp_x(i, j, 1), qp_x(i, j, 2)};
            Array_3 tmp = {
                qp_x0(i, j, 0) - x0_root[0], qp_x0(i, j, 1) - x0_root[1],
                qp_x0(i, j, 2) - x0_root[2]};
            tmp = RotateVectorByQuaternion(r_u, tmp);
            Array_3 x_undef{tmp[0] + x_root[0], tmp[1] + x_root[1], tmp[2] + x_root[2]};
            Array_3 u{x[0] - x_undef[0], x[1] - x_undef[1], x[2] - x_undef[2]};
            deformation_vector->InsertNextTuple(u.data());
        }
    }
    gd->GetPointData()->AddArray(deformation_vector);

    // Write the file
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetFileName(filename.c_str());
    writer->SetInputData(gd);
    writer->SetDataModeToAscii();
    writer->Write();
}

}  // namespace openturbine::tests
