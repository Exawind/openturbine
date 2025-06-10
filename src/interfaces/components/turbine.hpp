#pragma once

#include "interfaces/components/beam.hpp"
#include "interfaces/components/turbine_input.hpp"
#include "interfaces/constraint_data.hpp"
#include "interfaces/host_state.hpp"
#include "interfaces/node_data.hpp"
#include "model/model.hpp"

namespace openturbine::interfaces::components {

//< Minimum valid hub diameter
static constexpr double kMinHubDiameter{1e-8};

/**
 * @brief Represents a turbine with nodes, elements, and constraints
 *
 * This class is responsible for creating and managing a turbine on input
 * specifications. It handles the creation of nodes, mass elements, and constraints
 * within the provided model.
 */
class Turbine {
public:
    //--------------------------------------------------------------------------
    // Elements
    //--------------------------------------------------------------------------

    std::vector<Beam> blades;                        //< Blades in the turbine
    Beam tower;                                      //< Tower in the turbine
    size_t hub_mass_element_id{kInvalidID};          //< Hub mass element ID
    size_t yaw_bearing_mass_element_id{kInvalidID};  //< Yaw bearing mass element ID

    //--------------------------------------------------------------------------
    // Nodes
    //--------------------------------------------------------------------------

    std::vector<NodeData> apex_nodes;  //< Blade root nodes
    NodeData hub_node;                 //< Hub node
    NodeData azimuth_node;             //< Azimuth node
    NodeData shaft_base_node;          //< Shaft base node
    NodeData yaw_bearing_node;         //< Yaw bearing node

    //--------------------------------------------------------------------------
    // Constraints
    //--------------------------------------------------------------------------

    ConstraintData tower_base;                 //< Tower base constraint
    ConstraintData tower_top_to_yaw_bearing;   //< Tower top to yaw bearing constraint
    ConstraintData yaw_bearing_to_shaft_base;  //< Yaw bearing to shaft base constraint
    ConstraintData shaft_base_to_azimuth;      //< Nacelle mass to generator constraint
    ConstraintData azimuth_to_hub;             //< Azimuth to hub constraint
    std::vector<ConstraintData> blade_pitch;   //< Blade root to apex constraints
    std::vector<ConstraintData> apex_to_hub;   //< Apex to hub constraints

    //--------------------------------------------------------------------------
    // Controls
    //--------------------------------------------------------------------------

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
        : blades(CreateBlades(input.blades, model)),
          tower(input.tower, model),
          hub_node(kInvalidID),
          azimuth_node(kInvalidID),
          shaft_base_node(kInvalidID),
          yaw_bearing_node(kInvalidID),
          tower_base(kInvalidID),
          tower_top_to_yaw_bearing(kInvalidID),
          yaw_bearing_to_shaft_base(kInvalidID),
          shaft_base_to_azimuth(kInvalidID),
          azimuth_to_hub(kInvalidID),
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
        if (input.hub_diameter <= kMinHubDiameter) {
            throw std::invalid_argument(
                "Hub diameter must be > " + std::to_string(kMinHubDiameter) +
                ", got: " + std::to_string(input.hub_diameter)
            );
        }

        if (input.rotor_speed < 0.) {
            throw std::invalid_argument(
                "rotor speed must be >= 0, got: " + std::to_string(input.rotor_speed)
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

        // Calculate hub position based on tower top and apex offset
        const auto& tower_top_node = model.GetNode(this->tower.nodes.back().id);
        const Array_3 apex_position{
            tower_top_node.x0[0] - input.tower_axis_to_rotor_apex,
            tower_top_node.x0[1],
            tower_top_node.x0[2] + input.tower_top_to_rotor_apex,
        };

        // Calculate angle between blades
        const double blade_angle_delta = 2. * M_PI / static_cast<double>(this->blades.size());

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
            model.AddNode().SetPosition(tower_top_node.x0).SetOrientation({1., 0., 0., 0.}).Build()
        );

        // Create vector of rotor node IDs (hub, azimuth, shaft base)
        std::vector<size_t> rotor_node_ids{
            this->hub_node.id, this->azimuth_node.id, this->shaft_base_node.id
        };

        // Loop over blades
        for (auto i = 0U; i < this->blades.size(); ++i) {
            // Get node IDs for this blade
            std::vector<size_t> node_ids;
            std::transform(
                this->blades[i].nodes.begin(), this->blades[i].nodes.end(),
                std::back_inserter(node_ids),
                [](const NodeData& node) {
                    return node.id;
                }
            );

            // Loop through node IDs and rotate them to align with global Z-axis
            // then translate to hub radius so blade root is at the hub radius
            for (const auto& node_id : node_ids) {
                model.GetNode(node_id)
                    .RotateAboutPoint(q_x_to_z, origin)
                    .Translate({0., 0., input.hub_diameter / 2.});
            }

            // Add blade apex node at the origin and add node ID to node_ids
            const auto apex_node_id =
                model.AddNode().SetPosition({0., 0., 0., 1., 0., 0., 0.}).Build();
            node_ids.push_back(apex_node_id);

            // Add apex node to the model and rotor node IDs
            this->apex_nodes.emplace_back(apex_node_id);

            // Calculate cone angle rotation quaternion
            const auto q_cone = RotationVectorToQuaternion({0., -input.cone_angle, 0.});

            // Calculate azimuth angle rotation quaternion
            const auto q_azimuth =
                RotationVectorToQuaternion({static_cast<double>(i) * blade_angle_delta, 0., 0.});

            // Rotate blade nodes and apex node to cone angle and then to azimuth angle
            for (const auto& node_id : node_ids) {
                model.GetNode(node_id)
                    .RotateAboutPoint(q_cone, origin)
                    .RotateAboutPoint(q_azimuth, origin);
            }

            // Add blade and apex node IDs to rotor node IDs
            rotor_node_ids.insert(rotor_node_ids.end(), node_ids.begin(), node_ids.end());
        }

        // Create shaft rotation quaternion
        const auto q_shaft_tilt = RotationVectorToQuaternion({0., input.shaft_tilt_angle, 0.});

        // Rotate rotor nodes by shaft tilt angle and translate to apex position
        for (const auto& node_id : rotor_node_ids) {
            model.GetNode(node_id).RotateAboutPoint(q_shaft_tilt, origin).Translate(apex_position);
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
                root_node.x0[0] - apex_node.x0[0],
                root_node.x0[1] - apex_node.x0[1],
                root_node.x0[2] - apex_node.x0[2],
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
    [[nodiscard]] static std::vector<Beam> CreateBlades(
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
