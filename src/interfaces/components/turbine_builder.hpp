#pragma once

#include "interfaces/components/beam_builder.hpp"
#include "interfaces/components/turbine.hpp"
#include "interfaces/components/turbine_input.hpp"

namespace openturbine::interfaces::components {

/**
 * @brief Builder class for creating Turbine objects with a fluent interface pattern
 */
class TurbineBuilder {
public:
    TurbineBuilder() = default;

    /**
     * @brief Get the current blade input configuration
     * @return Reference to the current blade input
     */
    [[nodiscard]] const TurbineInput& Input() const { return this->input; }

    /**
     * @brief Build a Blade object from the current configuration
     * @param model The model to associate with this blade
     * @return A new Blade object
     */
    [[nodiscard]] Turbine Build(Model& model) const { return {this->input, model}; }

    /**
     * @brief Get reference to builder for a specific blade
     * @param n The index of the blade
     * @return Reference to the blade builder
     */
    [[nodiscard]] components::BeamBuilder& Blade(size_t i) {
        const auto n = i + 1;
        if (n > this->blade_builders.size()) {
            this->blade_builders.resize(n);
        }
        return this->blade_builders[i];
    }

    /**
     * @brief Get reference to builder for the tower
     * @return Reference to the tower builder
     */
    [[nodiscard]] components::BeamBuilder& Tower() { return this->tower_builder; }

    TurbineBuilder& SetNacelleMassPosition(const std::array<double, 3>& position) {
        this->input.nacelle_mass_position = position;
        return *this;
    }

    /**
     * @brief Set the hub height above the tower top (meters)
     * @param height The hub height to set
     * @return Reference to the builder
     */
    TurbineBuilder& SetHubHeightFromTowerTop(double height) {
        this->input.tower_top_to_hub = height;
        return *this;
    }

    /**
     * @brief Set the distance from tower axis to hub (meters)
     * @param distance The distance to set
     * @return Reference to the builder
     */
    TurbineBuilder& SetTowerAxisToHub(double distance) {
        this->input.tower_axis_to_hub = distance;
        return *this;
    }

    /**
     * @brief Set the shaft tilt angle (degrees)
     * @param angle The shaft tilt angle to set
     * @return Reference to the builder
     */
    TurbineBuilder& SetShaftTiltAngle(double angle) {
        this->input.shaft_tilt_angle = angle;
        return *this;
    }

    /**
     * @brief Set the azimuth angle (degrees)
     * @param angle The azimuth angle to set
     * @return Reference to the builder
     */
    TurbineBuilder& SetAzimuthAngle(double angle) {
        this->input.azimuth_angle = angle;
        return *this;
    }

    /**
     * @brief Set the initial nacelle yaw angle (degrees)
     * @param angle The nacelle yaw angle to set
     * @return Reference to the builder
     */
    TurbineBuilder& SetNacelleYawAngle(double angle) {
        this->input.nacelle_yaw_angle = angle;
        return *this;
    }

    /**
     * @brief Set the initial blade pitch angle (degrees)
     * @param angle The blade pitch angle to set
     * @return Reference to the builder
     */
    TurbineBuilder& SetBladePitchAngle(double angle) {
        this->input.blade_pitch_angle = angle;
        return *this;
    }

    /**
     * @brief Set the initial rotor speed (RPM)
     * @param speed The rotor speed to set
     * @return Reference to the builder
     */
    TurbineBuilder& SetRotorSpeed(double speed) {
        this->input.rotor_speed = speed;
        return *this;
    }

private:
    TurbineInput input;
    std::vector<BeamBuilder> blade_builders;  ///< Builders for the Blade components
    BeamBuilder tower_builder;                ///< Builder for the Tower component
};

}  // namespace openturbine::interfaces::components
