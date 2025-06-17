#pragma once

#include "interfaces/components/beam_builder.hpp"
#include "interfaces/components/turbine.hpp"
#include "interfaces/components/turbine_input.hpp"

namespace openturbine::interfaces::components {

/// @brief Builder class for creating Turbine objects with a fluent interface pattern
class TurbineBuilder {
public:
    TurbineBuilder() = default;

    /**
     * @brief Get the current turbine input configuration
     * @return Reference to the current turbine input
     */
    [[nodiscard]] const TurbineInput& Input() {
        // Add the blade inputs from the blade builders
        this->input.blades.clear();
        for (const auto& builder : this->blade_builders) {
            this->input.blades.push_back(builder.Input());
        }

        // Get the tower input from the tower builder
        this->input.tower = this->tower_builder.Input();

        return input;
    }

    /**
     * @brief Build a Turbine object from the current configuration
     * @param model The model to associate with this turbine
     * @return A new Turbine object
     */
    [[nodiscard]] Turbine Build(Model& model) {
        // build turbine and return
        return {this->Input(), model};
    }

    //--------------------------------------------------------------------------
    // Build components
    //--------------------------------------------------------------------------

    /**
     * @brief Get reference to builder for a specific blade
     * @param blade_index The index of the blade
     * @return Reference to the blade builder
     */
    [[nodiscard]] components::BeamBuilder& Blade(size_t blade_index) {
        // Ensure we have enough blade builders
        if (blade_index >= this->blade_builders.size()) {
            this->blade_builders.resize(blade_index + 1);
        }
        return this->blade_builders[blade_index];
    }

    /**
     * @brief Get reference to builder for the tower
     * @return Reference to the tower builder
     */
    [[nodiscard]] components::BeamBuilder& Tower() {
        // return tower builder
        return this->tower_builder;
    }

    /**
     * @brief Set the yaw bearing inertia matrix (6x6)
     * @param matrix The inertia matrix to set, includes yaw bearing and nacelle mass with inertia
     * about yaw bearing i.e. system_inertia_tt in WindIO
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetYawBearingInertiaMatrix(const std::array<std::array<double, 6>, 6>& matrix) {
        this->input.yaw_bearing_inertia_matrix = matrix;
        return *this;
    }

    /**
     * @brief Set the hub inertia matrix (6x6)
     * @param matrix The inertia matrix to set, includes hub assembly mass and inertia
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetHubInertiaMatrix(const std::array<std::array<double, 6>, 6>& matrix) {
        this->input.hub_inertia_matrix = matrix;
        return *this;
    }

    //--------------------------------------------------------------------------
    // Build geometric configuration of the turbine
    //--------------------------------------------------------------------------

    /**
     * @brief Set the position of the tower base node
     * @param position Array containing position/orientation [x,y,z,qw,qx,qy,qz]
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetTowerBasePosition(const std::array<double, 7>& position) {
        this->input.tower_base_position = position;
        return *this;
    }

    /**
     * @brief Set the distance from tower axis to hub i.e. distance from tower axis -> rotor apex
     * (meters)
     * @param distance The distance to set
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetTowerAxisToRotorApex(double distance) {
        this->input.tower_axis_to_rotor_apex = distance;
        return *this;
    }

    /**
     * @brief Set the hub height above the tower top i.e. distrance from tower top -> rotor apex
     * (meters)
     * @param height The hub height to set
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetTowerTopToRotorApex(double height) {
        this->input.tower_top_to_rotor_apex = height;
        return *this;
    }

    /**
     * @brief Distance from rotor apex to hub center of mass (meters)
     * @param distance The distance to set (meters)
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetRotorApexToHub(double distance) {
        this->input.rotor_apex_to_hub = distance;
        return *this;
    }

    /**
     * @brief Set the hub diameter (meters)
     * @param diameter The hub diameter to set (meters)
     * @return Reference to this builder for method chaining
     */
    TurbineBuilder& SetHubDiameter(double diameter) {
        this->input.hub_diameter = diameter;
        return *this;
    }

    //--------------------------------------------------------------------------
    // Build initial operating conditions of the turbine
    //--------------------------------------------------------------------------

    /**
     * @brief Set the shaft tilt angle (degrees)
     * @param angle The shaft tilt angle to set
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetShaftTiltAngle(double angle) {
        this->input.shaft_tilt_angle = angle;
        return *this;
    }

    /**
     * @brief Set the initial nacelle yaw angle (degrees)
     * @param angle The nacelle yaw angle to set
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetNacelleYawAngle(double angle) {
        this->input.nacelle_yaw_angle = angle;
        return *this;
    }

    /**
     * @brief Set the initial cone angle (radians)
     * @param angle The cone angle to set
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetConeAngle(double angle) {
        this->input.cone_angle = angle;
        return *this;
    }

    /**
     * @brief Set the initial blade pitch angle (degrees)
     * @param angle The blade pitch angle to set
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetBladePitchAngle(double angle) {
        this->input.blade_pitch_angle = angle;
        return *this;
    }

    /**
     * @brief Set the azimuth angle (degrees)
     * @param angle The azimuth angle to set
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetAzimuthAngle(double angle) {
        this->input.azimuth_angle = angle;
        return *this;
    }

    /**
     * @brief Set the initial rotor speed (rad/s)
     * @param speed The rotor speed to set
     * @return Reference to the builder for method chaining
     */
    TurbineBuilder& SetRotorSpeed(double speed) {
        this->input.rotor_speed = speed;
        return *this;
    }

private:
    TurbineInput input;                       ///< turbine configuration being built
    std::vector<BeamBuilder> blade_builders;  ///< builders for the blade components
    BeamBuilder tower_builder;                ///< builder for the tower component
};

}  // namespace openturbine::interfaces::components
