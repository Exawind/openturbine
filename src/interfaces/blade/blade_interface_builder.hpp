#pragma once

#include "blade_interface.hpp"
#include "interfaces/components/beam_builder.hpp"
#include "interfaces/components/solution_builder.hpp"

namespace openturbine::interfaces {

/**
 * @brief Builder class to construct a BladeInterface by composing Blade and Solution components
 *
 * @details This class combines these builders through a facade pattern, providing a unified
 * interface for constructing a complete blade model while keeping the configuration of
 * individual components separate and maintainable.
 *
 * - SolutionBuilder: Configures solver type, tolerances, time steps, and output settings
 * - BladeBuilder: Configures blade geometry, reference axes, section properties, and structural
 *                 matrices
 */
class BladeInterfaceBuilder {
public:
    [[nodiscard]] components::SolutionBuilder& Solution() { return this->solution_builder; }

    [[nodiscard]] components::BeamBuilder& Blade() { return this->beam_builder; }

    /**
     * @brief Builds the BladeInterface by composing the Blade and Solution components
     * @return A BladeInterface object
     */
    [[nodiscard]] BladeInterface Build() const {
        return BladeInterface(this->solution_builder.Input(), this->beam_builder.Input());
    }

private:
    components::SolutionBuilder solution_builder;  ///< Builder for the Solution component
    components::BeamBuilder beam_builder;          ///< Builder for the Blade component
};

}  // namespace openturbine::interfaces
