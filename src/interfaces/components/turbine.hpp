#pragma once

#include "interfaces/components/beam.hpp"
#include "interfaces/components/turbine_input.hpp"
#include "interfaces/constraint_data.hpp"
#include "interfaces/host_state.hpp"
#include "interfaces/node_data.hpp"
#include "model/model.hpp"

namespace openturbine::interfaces::components {

/// Translate node by a displacement vector
Array_7 TranslatePoint(const Array_7& point, const Array_3& displacement) {
    return {
        point[0] + displacement[0],
        point[1] + displacement[1],
        point[2] + displacement[2],
        point[3],  // Orientation remains unchanged
        point[4],
        point[5],
        point[6]
    };
}

/// Rotate node by a quaternion about the given point
Array_7 RotatePointAboutLoc(const Array_7& point, const Array_4& q, const Array_3& loc) {
    // Rotate point
    auto x_new =
        RotateVectorByQuaternion(q, {point[0] - loc[0], point[1] - loc[1], point[2] - loc[2]});

    // Rotate orientation
    auto q_new = QuaternionCompose(q, {point[3], point[4], point[5], point[6]});

    return {x_new[0] + loc[0], x_new[1] + loc[1], x_new[2] + loc[2], q_new[0],
            q_new[1],          q_new[2],          q_new[3]};
}

/// Rotate node by a rotation vector about the given point
Array_7 RotatePointAboutLoc(const Array_7& point, const Array_3& rv, const Array_3& loc) {
    return RotatePointAboutLoc(point, RotationVectorToQuaternion(rv), loc);
}

Array_7 RotatePoint(const Array_7& point, const Array_4& q) {
    // Rotate orientation
    auto q_new = QuaternionCompose(q, {point[3], point[4], point[5], point[6]});
    return {point[0], point[1], point[2], q_new[0], q_new[1], q_new[2], q_new[3]};
}

/**
 * @brief Represents a turbine with nodes, elements, and constraints
 *
 * This class is responsible for creating and managing a turbine on input
 * specifications. It handles the creation of nodes, mass elements, and constraints
 * within the provided model.
 */
class Turbine {
public:
    std::vector<Beam> blades;                        //< Blades in the turbine
    Beam tower;                                      //< Tower in the turbine
    size_t hub_mass_element_id{kInvalidID};          //< Hub mass element ID
    size_t yaw_bearing_mass_element_id{kInvalidID};  //< Yaw bearing mass element ID

    std::vector<NodeData> apex_nodes;  //< Blade root nodes
    NodeData hub_node;                 //< Hub node
    NodeData azimuth_node;             //< Azimuth node
    NodeData shaft_base_node;          //< Shaft base node
    NodeData yaw_bearing_node;         //< Yaw bearing node

    ConstraintData tower_base;                 //< Tower base constraint
    ConstraintData tower_top_to_yaw_bearing;   //< Tower top to yaw bearing constraint
    ConstraintData yaw_bearing_to_shaft_base;  //< Yaw bearing to shaft base constraint
    ConstraintData shaft_base_to_azimuth;      //< Nacelle mass to generator constraint
    ConstraintData azimuth_to_hub;             //< Azimuth to hub constraint
    std::vector<ConstraintData> blade_pitch;   //< Blade root to apex constraints
    std::vector<ConstraintData> apex_to_hub;   //< Apex to hub constraints

    std::vector<double> blade_pitch_control;  //< Blade pitch angles
    double torque_control{0.};                //< Torque control value
    double yaw_control{0.};                   //< Yaw control value

    /**
     * @brief
     * @param input Configuration for the turbine
     * @param model Model to which the turbine elements and nodes will be added
     * @throws std::invalid_argument If the input configuration is invalid
     */
    Turbine(const TurbineInput& input, Model& model)
        : blades(create_blades(input.blades, model)),
          tower(input.tower, model),
          apex_nodes(),
          hub_node(kInvalidID),
          azimuth_node(kInvalidID),
          shaft_base_node(kInvalidID),
          yaw_bearing_node(kInvalidID),
          tower_base(kInvalidID),
          tower_top_to_yaw_bearing(kInvalidID),
          yaw_bearing_to_shaft_base(kInvalidID),
          shaft_base_to_azimuth(kInvalidID),
          azimuth_to_hub(kInvalidID),
          blade_pitch(),
          blade_pitch_control(input.blades.size(), input.blade_pitch_angle) {
        // Validate turbine input
        ValidateInput(input);

        // Add intermediate nodes and position them to reference locations
        PositionNodes(input, model);

        // Constraints must be added after all nodes are positioned because
        // they depend on the initial positions of the nodes.
        AddConstraints(input, model);

        // SetInitialConditions(input, model);
    }

    /// @brief Populate node motion from host state
    /// @param host_state Host state containing position, displacement, velocity, and acceleration
    template <typename DeviceType>
    void GetMotion(const HostState<DeviceType>& host_state) {
        for (auto& blade : this->blades) {
            blade.GetMotion(host_state);
        }
        this->tower.GetMotion(host_state);
    }

    /// @brief Update the host state with current node forces and moments
    /// @param host_state Host state to update
    template <typename DeviceType>
    void SetLoads(HostState<DeviceType>& host_state) const {
        for (const auto& blade : this->blades) {
            blade.SetLoads(host_state);
        }
        this->tower.SetLoads(host_state);
    }

private:
    using DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

    /**
     * @brief ValidateInput
     */
    static void ValidateInput(const TurbineInput& input) {
        if (input.hub_diameter <= 1e-8) {
            throw std::invalid_argument(
                "Invalid hub diameter: " + std::to_string(input.hub_diameter)
            );
        }
    }

    /**
     * @brief Position nodes based on the turbine input configuration
     */
    void PositionNodes(const TurbineInput& input, Model& model) {
        // Define origin point for rotations
        const Array_3 origin{0., 0., 0.};

        // Calculate shaft length from tower top to hub and overhang
        const auto shaft_length = input.tower_axis_to_rotor_apex / cos(input.shaft_tilt_angle);

        // Rotate tower to align with global Z-axis
        const auto q_x_to_z = RotationVectorToQuaternion({0., -M_PI / 2., 0.});
        model.RotateBeamAboutPoint(this->tower.beam_element_id, q_x_to_z, origin);

        // Get tower top position from the last tower node
        const auto& tower_top_node = model.GetNode(this->tower.nodes.back().id);
        const Array_3 tower_top_position{
            tower_top_node.x[0],
            tower_top_node.x[1],
            tower_top_node.x[2],
        };

        // Calculate hub position based on tower top and hub offset
        const Array_3 apex_position{
            tower_top_position[0] - input.tower_axis_to_rotor_apex,
            tower_top_position[1],
            tower_top_position[2] + input.tower_top_to_rotor_apex,
        };

        // Calculate angle between blades
        const double blade_angle_delta = 2. * M_PI / this->blades.size();

        // Create hub node at origin
        this->hub_node = NodeData(
            model.AddNode().SetPosition({-input.rotor_apex_to_hub, 0., 0., 1., 0., 0., 0.}).Build()
        );

        // Create azimuth node at shaft base node
        this->azimuth_node =
            NodeData(model.AddNode().SetPosition({shaft_length, 0., 0., 1., 0., 0., 0.}).Build());

        // Create shaft base node relative to hub
        this->shaft_base_node =
            NodeData(model.AddNode().SetPosition({shaft_length, 0., 0., 1., 0., 0., 0.}).Build());

        // Create yaw bearing node at tower top position
        this->yaw_bearing_node = NodeData(
            model.AddNode().SetPosition(tower_top_node.x).SetOrientation({1., 0., 0., 0.}).Build()
        );

        // Create vector of rotor node IDs (hub, azimuth, shaft base, and blades)
        std::vector<size_t> rotor_node_ids{
            this->hub_node.id, this->azimuth_node.id, this->shaft_base_node.id
        };
        // Add all blade nodes to rotor_node_ids
        for (const auto& blade : this->blades) {
            std::transform(
                blade.nodes.begin(), blade.nodes.end(), std::back_inserter(rotor_node_ids),
                [](const NodeData& node) {
                    return node.id;
                }
            );
        }

        // Loop over blades
        for (auto i = 0U; i < this->blades.size(); ++i) {
            // Get element ID of the current blade
            const auto blade_element_id = this->blades[i].beam_element_id;

            // Rotate each blade to align with global Z-axis
            model.RotateBeamAboutPoint(blade_element_id, q_x_to_z, origin);

            // Translate blade by hub radius so blade root is at the hub radius
            model.TranslateBeam(blade_element_id, {0., 0., input.hub_diameter / 2.});

            // Add blade apex node at the origin
            const auto apex_node_id =
                model.AddNode().SetPosition({0., 0., 0., 1., 0., 0., 0.}).Build();
            this->apex_nodes.push_back(NodeData(apex_node_id));
            rotor_node_ids.push_back(apex_node_id);
            auto& apex_node = model.GetNode(apex_node_id);

            // Rotate blade for cone angle
            const auto q_cone = RotationVectorToQuaternion({0., -input.cone_angle, 0.});
            model.RotateBeamAboutPoint(blade_element_id, q_cone, origin);
            apex_node.RotateAboutPoint(q_cone, origin);

            // Rotate blade to azimuth angle
            const auto q_azimuth =
                RotationVectorToQuaternion({static_cast<double>(i) * blade_angle_delta, 0., 0.});
            model.RotateBeamAboutPoint(blade_element_id, q_azimuth, origin);
            apex_node.RotateAboutPoint(q_azimuth, origin);
        }

        // Create shaft rotation quaternion
        const auto shaft_rotation = RotationVectorToQuaternion({0., input.shaft_tilt_angle, 0.});

        // Rotate rotor nodes by shaft tilt angle
        for (const auto& node_id : rotor_node_ids) {
            model.GetNode(node_id).RotateAboutPoint(shaft_rotation, origin);
        }

        // Translate rotor to global position
        for (const auto& node_id : rotor_node_ids) {
            model.GetNode(node_id).Translate(apex_position);
        }
    }

    /**
     * @brief Add constraints
     */
    void AddConstraints(const TurbineInput& input, Model& model) {
        // Loop through blades
        for (auto i = 0U; i < this->blades.size(); ++i) {
            // Get the blade apex node
            const auto& apex_node = model.GetNode(this->apex_nodes[i].id);

            // Get the blade root node
            const auto& root_node = model.GetNode(this->blades[i].nodes[0].id);

            // Calculate the pitch axis for the blade
            const Array_3 pitch_axis{
                root_node.x[0] - apex_node.x[0],
                root_node.x[1] - apex_node.x[1],
                root_node.x[2] - apex_node.x[2],
            };

            // Create pitch control constraint
            this->blade_pitch.emplace_back(model.AddRotationControl(
                {root_node.id, apex_node.id}, pitch_axis, &this->blade_pitch_control[i]
            ));

            // Add rigid constraint between hub and blade apex
            this->apex_to_hub.emplace_back(
                model.AddRigidJointConstraint({this->hub_node.id, apex_node.id})
            );
        }

        // Add rigid constraint between hub and azimuth node
        this->azimuth_to_hub =
            ConstraintData(model.AddRigidJointConstraint({this->azimuth_node.id, this->hub_node.id})
            );

        // Shaft axis constraint
        const Array_3 shaft_axis{-cos(input.shaft_tilt_angle), 0., sin(input.shaft_tilt_angle)};
        this->shaft_base_to_azimuth = ConstraintData(model.AddRevoluteJointConstraint(
            {this->shaft_base_node.id, this->azimuth_node.id}, shaft_axis, &torque_control
        ));

        // Add constraint from yaw bearing to shaft base
        this->yaw_bearing_to_shaft_base = ConstraintData(
            model.AddRigidJointConstraint({this->yaw_bearing_node.id, this->shaft_base_node.id})
        );

        // Add constraint from tower top to yaw bearing
        this->tower_top_to_yaw_bearing = ConstraintData(model.AddRotationControl(
            {this->tower.nodes.back().id, this->yaw_bearing_node.id}, {0., 0., 1.},
            &this->yaw_control
        ));

        // Add fixed constraint at the tower base
        this->tower_base = ConstraintData(model.AddFixedBC(this->tower.nodes.front().id));
    }

    /**
     * @brief Initial displacements and velocities of the nodes
     */
    // void SetInitialConditions(const TurbineInput& input, Model& model) {}

    /// @brief  Create blades from input configuration
    /// @param blade_inputs Blade input configurations
    /// @param model Model to which the blades will be added
    /// @return Vector of blades
    [[nodiscard]] static std::vector<Beam> create_blades(
        const std::vector<BeamInput>& blade_inputs, Model& model
    ) {
        std::vector<Beam> blades;
        std::transform(
            blade_inputs.begin(), blade_inputs.end(), std::back_inserter(blades),
            [&model](const BeamInput& input) {
                return Beam(input, model);
            }
        );

        return blades;
    }
};

}  // namespace openturbine::interfaces::components
