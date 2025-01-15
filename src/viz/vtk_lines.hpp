#pragma once

#include <array>
#include <vector>

#ifdef OpenTurbine_ENABLE_VTK

#include <vtkCellType.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkLine.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>

#endif

namespace openturbine {

#ifdef OpenTurbine_ENABLE_VTK

inline void WriteLinesVTK(
    const std::vector<std::array<size_t, 2>>& lines,
    const std::vector<std::array<double, 7>>& points, const std::string& filename
) {
    // Create a grid object and add the points to it.
    const auto gd = vtkNew<vtkUnstructuredGrid>();
    gd->Allocate(static_cast<vtkIdType>(lines.size()));
    for (const auto& line : lines) {
        std::vector<vtkIdType> pts;
        pts.push_back(static_cast<vtkIdType>(line[0]));
        pts.push_back(static_cast<vtkIdType>(line[1]));
        gd->InsertNextCell(VTK_LINE, static_cast<vtkIdType>(2), pts.data());
    }

    // Set point positions
    const auto pt_pos = vtkNew<vtkPoints>();
    for (const auto& point : points) {
        pt_pos->InsertNextPoint(point[0], point[1], point[2]);
    }
    gd->SetPoints(pt_pos);

    // Write the file
    const auto writer = vtkNew<vtkXMLUnstructuredGridWriter>();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(gd);
    writer->SetDataModeToAscii();
    writer->Write();
}

#else

inline void
WriteLinesVTK(const std::vector<std::array<size_t, 2>>&, const std::vector<std::array<double, 7>>&, const std::string&) {
}

#endif

}  // namespace openturbine
