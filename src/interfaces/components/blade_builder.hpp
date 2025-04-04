#pragma once

#include "blade.hpp"
#include "blade_input.hpp"

namespace openturbine::interfaces::components {

struct BladeBuilder {
    BladeBuilder& SetElementOrder(size_t element_order) {
        input.element_order = element_order;
        return *this;
    }

    BladeBuilder& SetSectionRefinement(size_t section_refinement) {
        input.section_refinement = section_refinement;
        return *this;
    }

    BladeBuilder& ClearReferenceAxisPoints() {
        input.ref_axis.coordinate_grid.clear();
        input.ref_axis.coordinates.clear();
        return *this;
    }

    BladeBuilder& AddRefAxisPointX(double grid_location, const std::array<double, 3>& coordinates) {
        input.ref_axis.coordinate_grid.emplace_back(grid_location);
        input.ref_axis.coordinates.emplace_back(coordinates);
        return *this;
    }

    BladeBuilder& AddRefAxisPointZ(double grid_location, const std::array<double, 3>& coordinates) {
        const auto q_z_to_x = RotationVectorToQuaternion({0., M_PI / 2., 0.});
        input.ref_axis.coordinate_grid.emplace_back(grid_location);
        input.ref_axis.coordinates.emplace_back(RotateVectorByQuaternion(q_z_to_x, coordinates));
        return *this;
    }

    BladeBuilder& AddRefAxisTwist(double grid_location, double twist) {
        input.ref_axis.twist_grid.emplace_back(grid_location);
        input.ref_axis.twist.emplace_back(twist * M_PI / 180.);
        return *this;
    }

    BladeBuilder& PrescribedRootMotion(bool enable) {
        input.root.prescribe_root_motion = enable;
        return *this;
    }

    BladeBuilder& SetRootPosition(const std::array<double, 7>& p) {
        input.root.position = p;
        return *this;
    }

    BladeBuilder& SetRootVelocity(const std::array<double, 6>& v) {
        input.root.velocity = v;
        return *this;
    }

    BladeBuilder& SetRootAcceleration(const std::array<double, 6>& a) {
        input.root.acceleration = a;
        return *this;
    }

    BladeBuilder& ClearSections() {
        input.ref_axis.coordinate_grid.clear();
        input.ref_axis.coordinates.clear();
        return *this;
    }

    /// @brief Add sectional location, mass matrix, and stiffness matrix. Assumes that reference axis
    /// is along X
    /// @param grid_location section location [0,1] along blade
    /// @param mass_matrix sectional mass matrix
    /// @param stiffness_matrix sectional stiffness matrix
    /// @return
    BladeBuilder& AddSectionRefX(
        double grid_location, const Array_6x6& mass_matrix, const Array_6x6& stiffness_matrix
    ) {
        input.sections.emplace_back(grid_location, mass_matrix, stiffness_matrix);
        return *this;
    }

    /// @brief Add sectional location, mass matrix, and stiffness matrix. Assumes that reference axis
    /// is along Z
    /// @param grid_location section location [0,1] along blade
    /// @param mass_matrix sectional mass matrix
    /// @param stiffness_matrix sectional stiffness matrix
    /// @return
    BladeBuilder& AddSectionRefZ(
        double grid_location, const Array_6x6& mass_matrix, const Array_6x6& stiffness_matrix
    ) {
        const auto q_z_to_x = RotationVectorToQuaternion({0., -M_PI / 2., 0.});
        input.sections.emplace_back(
            grid_location, RotateMatrix6(mass_matrix, q_z_to_x),
            RotateMatrix6(stiffness_matrix, q_z_to_x)
        );
        return *this;
    }

    BladeBuilder& SetNodeSpacingLinear() {
        input.node_spacing = BladeInput::NodeSpacing::Linear;
        return *this;
    }

    BladeBuilder& SetNodeSpacingGaussLobattoLegendre() {
        input.node_spacing = BladeInput::NodeSpacing::GaussLobattoLegendre;
        return *this;
    }

    [[nodiscard]] Blade Build(Model& model) const { return {this->input, model}; }

    [[nodiscard]] const BladeInput& Input() const { return this->input; }

private:
    BladeInput input;
};

}  // namespace openturbine::interfaces::components
