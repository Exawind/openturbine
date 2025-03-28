#pragma once

#include <array>
#include <vector>

#include <vtkLine.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLPolyDataWriter.h>

namespace openturbine {

inline void WriteLinesVTK(
    const std::vector<std::array<size_t, 2>>& lines_data,
    const std::vector<std::array<double, 7>>& points, const std::string& filename
) {
    auto lines = vtkSmartPointer<vtkCellArray>::New();
    for (const auto& line_data : lines_data) {
        auto line = vtkSmartPointer<vtkLine>::New();
        line->GetPointIds()->SetId(0, static_cast<vtkIdType>(line_data[0]));
        line->GetPointIds()->SetId(1, static_cast<vtkIdType>(line_data[1]));
        lines->InsertNextCell(line);
    }

    // Set point positions
    auto pt_pos = vtkSmartPointer<vtkPoints>::New();
    for (const auto& point : points) {
        pt_pos->InsertNextPoint(
            static_cast<double>(static_cast<float>(point[0])),
            static_cast<double>(static_cast<float>(point[1])),
            static_cast<double>(static_cast<float>(point[2]))
        );
    }

    auto lines_poly_data = vtkSmartPointer<vtkPolyData>::New();
    lines_poly_data->SetPoints(pt_pos);
    lines_poly_data->SetLines(lines);

    // Write the file
    const auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName((filename + ".vtp").c_str());
    writer->SetInputData(lines_poly_data);
    writer->SetDataModeToAscii();
    writer->Write();
}

}  // namespace openturbine
