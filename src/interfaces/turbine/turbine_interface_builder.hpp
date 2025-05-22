#pragma once

#include "interfaces/components/beam_builder.hpp"
#include "interfaces/components/solution_builder.hpp"
#include "turbine_interface.hpp"

namespace openturbine::interfaces {

/**
 * @brief Builder class to construct a TurbineInterface by composing Turbine and Solution components
 *
 * @details This class combines these builders through a facade pattern, providing a unified
 * interface for constructing a complete blade model while keeping the configuration of
 * individual components separate and maintainable.
 *
 * - SolutionBuilder: Configures solver type, tolerances, time steps, and output settings
 * - TurbineBuilder: Configures blade geometry, reference axes, section properties, and structural
 *                 matrices
 */
class TurbineInterfaceBuilder {
public:
    /// @brief Gets the builder for the solution component
    /// @return A reference to the SolutionBuilder for the solution component
    [[nodiscard]] components::SolutionBuilder& Solution() { return this->solution_builder; }

    /// @brief Gets the builder for a specific blade
    /// @param n The index of the blade (first blade is 0)
    /// @return A reference to the BeamBuilder for the specified blade
    [[nodiscard]] components::BeamBuilder& Blade(size_t n) {
        if (n > this->blade_builders.size()) {
            this->blade_builders.resize(n);
        }
        return this->blade_builders.at(n);
    }

    /// @brief Gets the builder for the tower
    /// @return A reference to the BeamBuilder for the tower
    [[nodiscard]] components::BeamBuilder& Tower() { return this->tower_builder; }

    /**
     * @brief Builds the TurbineInterface by composing the blade, tower, nacelle, hub, and solution
     * components
     * @return A TurbineInterface object
     */
    [[nodiscard]] TurbineInterface Build() const {
        std::vector<components::BeamInput> blade_inputs;
        for (const auto& builder : this->blade_builders) {
            blade_inputs.push_back(builder.Input());
        }
        return TurbineInterface(
            this->solution_builder.Input(), blade_inputs, this->tower_builder.Input()
        );
    }

private:
    components::SolutionBuilder solution_builder;         ///< Builder for the Solution component
    std::vector<components::BeamBuilder> blade_builders;  ///< Builder for the blades components
    components::BeamBuilder tower_builder;                ///< Builder for the tower component
};

}  // namespace openturbine::interfaces
