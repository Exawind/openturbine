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

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/math/quaternion_operations.hpp"

namespace openturbine::tests {

inline std::vector<std::vector<double>> kokkos_view_2D_to_vector(Kokkos::View<double**> view) {
    Kokkos::View<double**> view_contiguous("view_contiguous", view.extent(0), view.extent(1));
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror(view_contiguous);
    Kokkos::deep_copy(view_host, view_contiguous);
    std::vector<std::vector<double>> values(view.extent(0));
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        for (size_t j = 0; j < view_host.extent(1); ++j) {
            values[i].push_back(view_host(i, j));
        }
    }
    return values;
}

inline std::vector<std::vector<std::vector<double>>> kokkos_view_3D_to_vector(
    Kokkos::View<double***> view
) {
    Kokkos::View<double***> view_contiguous(
        "view_contiguous", view.extent(0), view.extent(1), view.extent(2)
    );
    Kokkos::deep_copy(view_contiguous, view);
    auto view_host = Kokkos::create_mirror(view_contiguous);
    Kokkos::deep_copy(view_host, view_contiguous);
    std::vector<std::vector<std::vector<double>>> values(view.extent(0));
    for (size_t i = 0; i < view_host.extent(0); ++i) {
        for (size_t j = 0; j < view_host.extent(1); ++j) {
            values[i].push_back(std::vector<double>());
            for (size_t k = 0; k < view_host.extent(2); ++k) {
                values[i][j].push_back(view_host(i, j, k));
            }
        }
    }
    return values;
}

inline void BeamsWriteVTK(Beams& beams, std::string filename) {
    // Get a copy of the beam element indices in the host space
    auto elem_indices = Kokkos::create_mirror(beams.elem_indices);
    Kokkos::deep_copy(elem_indices, beams.elem_indices);

    // Create a grid object and add the points to it.
    vtkNew<vtkUnstructuredGrid> gd;
    gd->Allocate(beams.num_elems);
    for (int i = 0; i < beams.num_elems; ++i) {
        auto& idx = elem_indices[i];
        std::vector<vtkIdType> pts;
        pts.push_back(idx.qp_range.first);       // first qp
        pts.push_back(idx.qp_range.second - 1);  // last qp
        for (int j = idx.qp_range.first; j < idx.qp_range.second - 2; ++j) {
            pts.push_back(j);
        }
        gd->InsertNextCell(VTK_LAGRANGE_CURVE, idx.num_qps, pts.data());
    }

    // Create qp position points
    auto qp_x0 = kokkos_view_2D_to_vector(beams.qp_x0);
    auto qp_x = kokkos_view_2D_to_vector(beams.qp_u);
    for (size_t j = 0; j < qp_x.size(); ++j) {
        for (size_t k = 0; k < qp_x[0].size(); ++k) {
            qp_x[j][k] += qp_x0[j][k];
        }
    }
    vtkNew<vtkPoints> qp_pos;
    for (const auto& p : qp_x) {
        qp_pos->InsertNextPoint(p[0], p[1], p[2]);
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
    auto qp_r = kokkos_view_2D_to_vector(beams.qp_r);
    auto qp_r0 = kokkos_view_2D_to_vector(beams.qp_r0);
    for (size_t i = 0; i < qp_r.size(); ++i) {
        Array_4 r{qp_r[i][0], qp_r[i][1], qp_r[i][2], qp_r[i][3]};
        Array_4 r0{qp_r0[i][0], qp_r0[i][1], qp_r0[i][2], qp_r0[i][3]};
        auto R = QuaternionToRotationMatrix(QuaternionCompose(r, r0));
        double ori_x[3] = {R[0][0], R[1][0], R[2][0]};
        double ori_y[3] = {R[0][1], R[1][1], R[2][1]};
        double ori_z[3] = {R[0][2], R[1][2], R[2][2]};
        orientation_x->InsertNextTuple(ori_x);
        orientation_y->InsertNextTuple(ori_y);
        orientation_z->InsertNextTuple(ori_z);
    }
    gd->GetPointData()->AddArray(orientation_x);
    gd->GetPointData()->AddArray(orientation_y);
    gd->GetPointData()->AddArray(orientation_z);

    // Add translational velocity point data
    vtkNew<vtkFloatArray> translational_velocity;
    translational_velocity->SetNumberOfComponents(3);
    translational_velocity->SetName("TranslationalVelocity");
    auto qp_u_dot = kokkos_view_2D_to_vector(beams.qp_u_dot);
    for (const auto& p : qp_u_dot) {
        translational_velocity->InsertNextTuple(p.data());
    }
    gd->GetPointData()->AddArray(translational_velocity);

    // Add rotational velocity point data
    vtkNew<vtkFloatArray> rotational_velocity;
    rotational_velocity->SetNumberOfComponents(3);
    rotational_velocity->SetName("RotationalVelocity");
    auto qp_omega = kokkos_view_2D_to_vector(beams.qp_omega);
    for (const auto& p : qp_omega) {
        rotational_velocity->InsertNextTuple(p.data());
    }
    gd->GetPointData()->AddArray(rotational_velocity);

    // Add deformation point data
    vtkNew<vtkFloatArray> deformation_vector;
    deformation_vector->SetNumberOfComponents(3);
    deformation_vector->SetName("DeformationVector");
    for (int i = 0; i < beams.num_elems; ++i) {
        auto& idx = elem_indices[i];
        auto i_root = idx.qp_range.first;
        Array_3 x0_root{qp_x0[i_root][0], qp_x0[i_root][1], qp_x0[i_root][2]};
        Array_3 x_root{qp_x[i_root][0], qp_x[i_root][1], qp_x[i_root][2]};
        Array_4 r_u{qp_r[i_root][0], qp_r[i_root][1], qp_r[i_root][2], qp_r[i_root][3]};

        for (int j = idx.qp_range.first; j < idx.qp_range.second; ++j) {
            Array_3 x{qp_x[j][0], qp_x[j][1], qp_x[j][2]};
            Array_3 tmp = {
                qp_x0[j][0] - x0_root[0], qp_x0[j][1] - x0_root[1], qp_x0[j][2] - x0_root[2]};
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