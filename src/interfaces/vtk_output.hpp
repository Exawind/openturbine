#pragma once

#include <algorithm>
#include <filesystem>

#include "interfaces/node_data.hpp"
#include "types.hpp"

#ifdef OpenTurbine_ENABLE_VTK
#include "viz/vtk_beam.hpp"
#include "viz/vtk_nodes.hpp"
#endif

namespace openturbine::interfaces {

struct VTKOutput {
    /// @brief File index
    size_t file_index{0};

    /// @brief Output directory
    std::filesystem::path output_dir;

    /// @brief Base name for vtk file, will have "_#####.vtk" appended
    std::string file_name_template;

    /// @brief Node identifier in model
    bool active;

    explicit VTKOutput(const std::filesystem::path& vtk_output_path, bool clean_dir = true)
        : output_dir(vtk_output_path.has_parent_path() ? vtk_output_path.parent_path() : "."),
          file_name_template(vtk_output_path.filename().string()),
          active(!vtk_output_path.empty()) {
        // If not active, return
        if (!active) {
            return;
        }

#ifndef OpenTurbine_ENABLE_VTK
        throw("VTK output requested, but VTK not enabled");
#endif

        // Create output directory if it does not exist
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

        // If directory cleaning requested, delete all files in output directory
        if (clean_dir) {
            for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
                if (entry.is_regular_file()) {
                    std::filesystem::remove(entry.path());
                }
            }
        }
    }

    void WriteNodes([[maybe_unused]] const std::vector<NodeData>& nodes) const {
        if (this->active) {
#ifdef OpenTurbine_ENABLE_VTK
            WriteNodesVTK(nodes, this->BuildFilePath() + ".vtp");
#endif
        }
    }

    void WriteBeam([[maybe_unused]] const std::vector<NodeData>& nodes) const {
        if (this->active) {
#ifdef OpenTurbine_ENABLE_VTK
            WriteBeamVTK(nodes, this->BuildFilePath() + ".vtu");
#endif
        }
    }

    [[nodiscard]] std::string BuildFilePath() const {
        const auto index_str = std::to_string(this->file_index);
        auto file_name{this->file_name_template};
        const auto count = std::count(file_name.begin(), file_name.end(), '#');
        const size_t n = count > 0 ? static_cast<size_t>(count) : 0U;
        const auto replace_str = std::string(n - index_str.size(), '0') + index_str;
        const auto pos = file_name.find(std::string(n, '#'));
        if (pos != std::string::npos) {
            file_name.replace(pos, n, replace_str);
        }
        return this->output_dir / file_name;
    }

    void IncrementFileIndex() { ++this->file_index; }
};

}  // namespace openturbine::interfaces
