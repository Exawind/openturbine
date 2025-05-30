#pragma once

#include "interfaces/components/beam.hpp"
#include "interfaces/components/turbine_input.hpp"
#include "interfaces/constraint_data.hpp"
#include "interfaces/host_state.hpp"
#include "interfaces/node_data.hpp"
#include "model/model.hpp"

namespace openturbine::interfaces::components {

/**
 * @brief Represents a turbine with nodes, elements, and constraints
 *
 * This class is responsible for creating and managing a turbine on input
 * specifications. It handles the creation of nodes, mass elements, and constraints
 * within the provided model.
 */
class Turbine {
public:
    std::vector<Beam> blades;  //< Blades in the turbine
    Beam tower;                //< Tower in the turbine

    NodeData nacelle_mass_node;  //< Nacelle center-of-mass node
    NodeData yaw_bearing_node;   //< Yaw bearing node
    NodeData shaft_base_node;    //< Shaft base node
    NodeData azimuth_node;       //< Azimuth node
    NodeData hub_node;           //< Hub node

    ConstraintData tower_base;                   //< Tower base constraint
    ConstraintData tower_top_to_yaw_bearing;     //< Tower top to yaw bearing constraint
    ConstraintData yaw_bearing_to_nacelle_mass;  //< Yaw bearing to nacelle mass constraint
    ConstraintData yaw_bearing_to_shaft_base;    //< Yaw bearing to shaft base constraint
    ConstraintData shaft_base_to_azimuth;        //< Nacelle mass to generator constraint
    ConstraintData azimuth_to_hub;               //< Azimuth to hub constraint
    std::vector<ConstraintData> hub_to_blades;   //< Hub to blade constraints

    /**
     * @brief
     * @param input Configuration for the turbine
     * @param model Model to which the turbine elements and nodes will be added
     * @throws std::invalid_argument If the input configuration is invalid
     */
    Turbine(const TurbineInput& input, Model& model)
        : blades(create_blades(input.blade_inputs, model)),
          tower(input.tower_input, model),
          nacelle_mass_node(kInvalidID),
          yaw_bearing_node(kInvalidID),
          shaft_base_node(kInvalidID),
          azimuth_node(kInvalidID),
          hub_node(kInvalidID),
          tower_base(kInvalidID),
          tower_top_to_yaw_bearing(kInvalidID),
          yaw_bearing_to_nacelle_mass(kInvalidID),
          yaw_bearing_to_shaft_base(kInvalidID),
          shaft_base_to_azimuth(kInvalidID),
          azimuth_to_hub(kInvalidID),
          hub_to_blades() {
        // Initialize blades from blade_inputs
        blades.reserve(input.blade_inputs.size());
        for (const auto& blade_input : input.blade_inputs) {
            blades.emplace_back(blade_input, model);
        }
    }

    /// @brief Populate node motion based on host state
    /// @param host_state Host state containing position, displacement, velocity, and acceleration
    template <typename DeviceType>
    void UpdateNodeMotionFromState(const HostState<DeviceType>& host_state) {
        for (auto& blade : blades) {
            for (auto& node : blade.nodes) {
                node.UpdateMotion(host_state);
            }
        }
        for (auto& node : tower.nodes) {
            node.UpdateMotion(host_state);
        }
    }

    /// @brief Update the host state with current node forces and moments
    /// @param host_state Host state to update
    template <typename DeviceType>
    void UpdateHostStateExternalLoads(HostState<DeviceType>& host_state) {
        for (const auto& blade : blades) {
            for (const auto& node : blade.nodes) {
                node.UpdateHostStateExternalLoads(host_state);
            }
        }
        for (auto& node : tower.nodes) {
            node.UpdateHostStateExternalLoads(host_state);
        }
    }

private:
    using DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

    /// @brief  Create blades from input configuration
    /// @param blade_inputs Blade input configurations
    /// @param model Model to which the blades will be added
    /// @return Vector of blades
    [[nodiscard]] static std::vector<Beam> create_blades(
        const std::vector<BeamInput>& blade_inputs, Model& model
    ) {
        std::vector<Beam> blades;
        for (const auto& input : blade_inputs) {
            blades.emplace_back(input, model);
        }
        return blades;
    }
};

}  // namespace openturbine::interfaces::components
