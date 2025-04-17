#pragma once

#include "blade.hpp"
#include "blade_input.hpp"

namespace openturbine::interfaces::components {

/**
 * @brief Enum to represent reference axis orientation
 */
enum class ReferenceAxisOrientation {
    X,  // X-axis
    Z   // Z-axis
};

/**
 * @brief Builder class for creating Blade objects with a fluent interface pattern
 */
class BladeBuilder {
public:
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

    /**
     * @brief Adds a reference axis point with specified orientation
     * @param grid_location Location [0,1] along the blade
     * @param coordinates Coordinates (x,y,z) of the reference axis point
     * @param ref_axis Reference axis orientation ('X' or 'Z')
     * @return Reference to this builder for method chaining
     */
    BladeBuilder& AddRefAxisPoint(
        double grid_location, const std::array<double, 3>& coordinates,
        ReferenceAxisOrientation ref_axis
    ) {
        input.ref_axis.coordinate_grid.emplace_back(grid_location);
        if (ref_axis == ReferenceAxisOrientation::X) {
            input.ref_axis.coordinates.emplace_back(coordinates);
            return *this;
        }
        if (ref_axis == ReferenceAxisOrientation::Z) {
            const auto q_z_to_x = RotationVectorToQuaternion({0., M_PI / 2., 0.});
            input.ref_axis.coordinates.emplace_back(RotateVectorByQuaternion(q_z_to_x, coordinates));
            return *this;
        }
        throw std::invalid_argument("Invalid reference axis orientation");
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

    /// @brief Adds sectional location, mass matrix, and stiffness matrix. Can handle reference
    /// axis along X or Z.
    /// @param grid_location section location [0,1] along blade
    /// @param mass_matrix sectional mass matrix
    /// @param stiffness_matrix sectional stiffness matrix
    /// @param ref_axis Reference axis ('X' or 'Z')
    /// @return Reference to this builder for method chaining
    BladeBuilder& AddSection(
        double grid_location, const Array_6x6& mass_matrix, const Array_6x6& stiffness_matrix,
        ReferenceAxisOrientation ref_axis
    ) {
        if (ref_axis == ReferenceAxisOrientation::X) {
            input.sections.emplace_back(grid_location, mass_matrix, stiffness_matrix);
            return *this;
        }
        if (ref_axis == ReferenceAxisOrientation::Z) {
            const auto q_z_to_x = RotationVectorToQuaternion({0., -M_PI / 2., 0.});
            input.sections.emplace_back(
                grid_location, RotateMatrix6(mass_matrix, q_z_to_x),
                RotateMatrix6(stiffness_matrix, q_z_to_x)
            );
            return *this;
        }
        throw std::invalid_argument("Invalid reference axis orientation");
    }

    /**
     * @brief Get the current blade input configuration
     * @return Reference to the current blade input
     */
    [[nodiscard]] const BladeInput& Input() const { return this->input; }

    /**
     * @brief Build a Blade object from the current configuration
     * @param model The model to associate with this blade
     * @return A new Blade object
     */
    [[nodiscard]] Blade Build(Model& model) const { return {this->input, model}; }

private:
    BladeInput input;
};

}  // namespace openturbine::interfaces::components
