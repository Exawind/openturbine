#pragma once

#include "beam_input.hpp"

namespace openturbine {
    class Model;
}

namespace openturbine::interfaces::components {

class Beam;

/**
 * @brief Enum to represent reference axis orientation
 */
enum class ReferenceAxisOrientation : std::uint8_t {
    X,  // X-axis
    Z   // Z-axis
};

/**
 * @brief Builder class for creating Blade objects with a fluent interface pattern
 */
class BeamBuilder {
public:
    BeamBuilder& SetElementOrder(size_t element_order);

    BeamBuilder& SetSectionRefinement(size_t section_refinement);

    BeamBuilder& ClearReferenceAxisPoints();

    /**
     * @brief Adds a reference axis point with specified orientation
     * @param grid_location Location [0,1] along the blade
     * @param coordinates Coordinates (x,y,z) of the reference axis point
     * @param ref_axis Reference axis orientation ('X' or 'Z')
     * @return Reference to this builder for method chaining
     */
    BeamBuilder& AddRefAxisPoint(
        double grid_location, const std::array<double, 3>& coordinates,
        ReferenceAxisOrientation ref_axis
    );

    BeamBuilder& AddRefAxisTwist(double grid_location, double twist);

    BeamBuilder& PrescribedRootMotion(bool enable);

    BeamBuilder& SetRootPosition(const std::array<double, 7>& p);

    BeamBuilder& SetRootVelocity(const std::array<double, 6>& v);

    BeamBuilder& SetRootAcceleration(const std::array<double, 6>& a);

    BeamBuilder& ClearSections();
    /// @brief Adds sectional location, mass matrix, and stiffness matrix. Can handle reference
    /// axis along X or Z.
    /// @param grid_location section location [0,1] along blade
    /// @param mass_matrix sectional mass matrix
    /// @param stiffness_matrix sectional stiffness matrix
    /// @param ref_axis Reference axis ('X' or 'Z')
    /// @return Reference to this builder for method chaining
    BeamBuilder& AddSection(
        double grid_location, const std::array<std::array<double, 6>, 6>& mass_matrix, const std::array<std::array<double, 6>, 6>& stiffness_matrix,
        ReferenceAxisOrientation ref_axis
    );

    /**
     * @brief Get the current blade input configuration
     * @return Reference to the current blade input
     */
    [[nodiscard]] const BeamInput& Input() const;

    /**
     * @brief Build a Blade object from the current configuration
     * @param model The model to associate with this blade
     * @return A new Blade object
     */
    [[nodiscard]] Beam Build(Model& model) const;

private:
    BeamInput input;
};

}  // namespace openturbine::interfaces::components
