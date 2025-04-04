#pragma once

#include <array>
#include <vector>

#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include "interfaces/node_data.hpp"
#include "math/quaternion_operations.hpp"

namespace openturbine::interfaces {

inline void WriteBeamVTK(const std::vector<NodeData>& nodes, const std::string& file_path) {
    const auto gd = vtkNew<vtkUnstructuredGrid>();
    gd->Allocate(1);

    std::vector<vtkIdType> pts;
    pts.push_back(0);
    pts.push_back(static_cast<vtkIdType>(nodes.size() - 1));
    for (size_t j = 1; j < nodes.size() - 1; ++j) {
        pts.push_back(static_cast<vtkIdType>(j));
    }
    gd->InsertNextCell(VTK_LAGRANGE_CURVE, static_cast<vtkIdType>(nodes.size()), pts.data());

    //--------------------------------------------------------------------------
    // Points
    //--------------------------------------------------------------------------

    auto points = vtkSmartPointer<vtkPoints>::New();
    for (const auto& node : nodes) {
        points->InsertNextPoint(node.position[0], node.position[1], node.position[2]);
    }

    gd->SetPoints(points);

    //--------------------------------------------------------------------------
    // Orientation data
    //--------------------------------------------------------------------------

    const auto orientation_x = vtkNew<vtkDoubleArray>();
    orientation_x->SetNumberOfComponents(3);
    orientation_x->SetName("OrientationX");
    const auto orientation_y = vtkNew<vtkDoubleArray>();
    orientation_y->SetNumberOfComponents(3);
    orientation_y->SetName("OrientationY");
    const auto orientation_z = vtkNew<vtkDoubleArray>();
    orientation_z->SetNumberOfComponents(3);
    orientation_z->SetName("OrientationZ");
    for (const auto& node : nodes) {
        auto R = QuaternionToRotationMatrix({
            node.position[3],
            node.position[4],
            node.position[5],
            node.position[6],
        });

        orientation_x->InsertNextValue(R[0][0]);
        orientation_x->InsertNextValue(R[1][0]);
        orientation_x->InsertNextValue(R[2][0]);

        orientation_y->InsertNextValue(R[0][1]);
        orientation_y->InsertNextValue(R[1][1]);
        orientation_y->InsertNextValue(R[2][1]);

        orientation_z->InsertNextValue(R[0][2]);
        orientation_z->InsertNextValue(R[1][2]);
        orientation_z->InsertNextValue(R[2][2]);
    }

    gd->GetPointData()->AddArray(orientation_x);
    gd->GetPointData()->AddArray(orientation_y);
    gd->GetPointData()->AddArray(orientation_z);

    //--------------------------------------------------------------------------
    // Velocity data
    //--------------------------------------------------------------------------

    // Add translational velocity point data
    const auto translational_velocity = vtkNew<vtkDoubleArray>();
    translational_velocity->SetNumberOfComponents(3);
    translational_velocity->SetName("TranslationalVelocity");
    for (const auto& node : nodes) {
        translational_velocity->InsertNextValue(node.velocity[0]);
        translational_velocity->InsertNextValue(node.velocity[1]);
        translational_velocity->InsertNextValue(node.velocity[2]);
    }
    gd->GetPointData()->AddArray(translational_velocity);

    const auto rotational_velocity = vtkNew<vtkDoubleArray>();
    rotational_velocity->SetNumberOfComponents(3);
    rotational_velocity->SetName("RotationalVelocity");
    for (const auto& node : nodes) {
        rotational_velocity->InsertNextValue(node.velocity[3]);
        rotational_velocity->InsertNextValue(node.velocity[4]);
        rotational_velocity->InsertNextValue(node.velocity[5]);
    }
    gd->GetPointData()->AddArray(rotational_velocity);

    //--------------------------------------------------------------------------
    // Acceleration data
    //--------------------------------------------------------------------------

    // Add translational acceleration point data
    const auto translational_acceleration = vtkNew<vtkDoubleArray>();
    translational_acceleration->SetNumberOfComponents(3);
    translational_acceleration->SetName("TranslationalAcceleration");
    for (const auto& node : nodes) {
        translational_acceleration->InsertNextValue(node.acceleration[0]);
        translational_acceleration->InsertNextValue(node.acceleration[1]);
        translational_acceleration->InsertNextValue(node.acceleration[2]);
    }
    gd->GetPointData()->AddArray(translational_acceleration);

    const auto rotational_acceleration = vtkNew<vtkDoubleArray>();
    rotational_acceleration->SetNumberOfComponents(3);
    rotational_acceleration->SetName("RotationalAcceleration");
    for (const auto& node : nodes) {
        rotational_acceleration->InsertNextValue(node.acceleration[3]);
        rotational_acceleration->InsertNextValue(node.acceleration[4]);
        rotational_acceleration->InsertNextValue(node.acceleration[5]);
    }
    gd->GetPointData()->AddArray(rotational_acceleration);

    //--------------------------------------------------------------------------
    // Write the file
    //--------------------------------------------------------------------------

    const auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    writer->SetFileName(file_path.c_str());
    writer->SetInputData(gd);
    writer->SetDataModeToAscii();
    writer->Write();
}

}  // namespace openturbine::interfaces
