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
 * This class is responsible for creating and managing a turbine based on input
 * specifications. It handles the creation of the beam elements, mass elements,
 * nodes, and constraints within the provided model.
 *
 * --------------------------------------------------------------------------
 * Node structure
 * --------------------------------------------------------------------------
 * The turbine assembly consists of multiple interconnected nodes that
 * represent different physical components and their kinematic relationships.
 *
 * Tower Nodes
 *  - Tower nodes: Beam nodes distributed along tower height (1, 2, ..., n)
 *  - Tower base: Fixed constraint point at tower foundation (first tower node)
 *  - Tower top: Connection point to nacelle assembly (last tower node)
 *
 *     ┌─── Tower top node (connection to nacelle)
 *     │
 *     ○ <- Tower node n
 *     │
 *     ○ <- Tower node n-1
 *     .
 *     .
 *     │
 *     ○ <- Tower node 2
 *     │
 *     ○ <- Tower node 1 (Tower base - fixed constraint)
 * -------------
 *  / / / /  <-  Ground / Foundation
 *
 * Nacelle/Drivetrain Nodes
 *  - Yaw bearing node: Located at tower top, allows nacelle yaw rotation
 *  - Shaft base node: Base of the main shaft within nacelle
 *  - Azimuth node: Intermediate node for rotor azimuth positioning
 *  - Hub node: Center of mass of the rotating hub assembly
 *
 *    Yaw bearing     Shaft base      Azimuth          Hub
 *        ●  ----------   ●  ----------  ●  ----------  ●
 *   (yaw control      (rigid         (torque        (rigid
 *    rotation        connection)      control)      connection
 *   about Z-axis)                                   to blades)
 *
 * Blade Assembly Nodes
 *  - Blade apex nodes: Connection points between hub and blade roots (one per blade)
 *  - Blade structural nodes: Beam nodes along each blade span
 *
 *                   pitch axis    |<----- Blade nodes --->|
 *     ● Blade apex -------------  ●  ------  ●  --------  ●
 *       node                   (root)                   (tip)
 *        │
 *        │ (rigid connection)
 *        │
 *        ● Hub node
 * --------------------------------------------------------------------------
 * Kinematic chain
 * --------------------------------------------------------------------------
 * The nodes are connected in a kinematic chain that represents the turbine's
 * degrees of freedom.
 *
 * Tower base node (fixed BC) -> Tower nodes -> Yaw Bearing node (yaw control,
 * rigid constraint) -> Shaft base node (torque control, rigid constraint) ->
 * Azimuth node (rigid constraint) -> Hub node (rigid constraint) -> Blade Apex
 * nodes (pitch control) -> Blade nodes
 */
class Turbine {
public:
    //--------------------------------------------------------------------------
    // Constants
    //--------------------------------------------------------------------------

    /// Minimum valid hub diameter
    static constexpr double kMinHubDiameter{1e-8};

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
    // Control inputs
    //--------------------------------------------------------------------------

    std::vector<double> blade_pitch_control;  //< Blade pitch angles
    double torque_control{0.};                //< Torque control value
    double yaw_control{0.};                   //< Yaw control value

    /**
     * @brief Constructs a turbine with the specified configuration
     *
     * Creates a complete turbine structure including blades, tower, hub, nacelle components,
     * and all associated nodes, constraints, and control systems. The turbine is positioned
     * and oriented according to the input parameters, with proper kinematic relationships
     * established between all components.
     * --------------------------------------------------------------------------
     * Construction sequence
     * --------------------------------------------------------------------------
     *  1. Validation: Input parameters are checked for physical consistency
     *  2. Node creation + positioning: All nodes are created and assembled in proper position
     *  3. Constraint creation: Kinematic relationships are established between nodes
     *  4. Initial conditions: Displacements/velocities are applied as necessary
     *
     * @param input Turbine configuration parameters
     * @param model Structural model to add turbine components to
     * @throws std::invalid_argument If input configuration is invalid
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
        // Validate turbine inputs
        ValidateInput(input);

        // Add intermediate nodes and position them to reference locations
        PositionNodes(input, model);

        // Constraints must be added after all nodes are positioned because
        // they depend on the initial positions of the nodes
        AddConstraints(input, model);

        // Set initial conditions (displacements and velocities) after positioning and constraints
        SetInitialConditions(input, model);
    }

    /**
     * @brief Populate node motion from host state
     * @param host_state Host state containing position, displacement, velocity, and acceleration
     */
    template <typename DeviceType>
    void GetMotion(const HostState<DeviceType>& host_state) {
        for (auto& blade : this->blades) {
            blade.GetMotion(host_state);
        }
        this->tower.GetMotion(host_state);
    }

    /**
     * @brief Update the host state with current node forces and moments
     * @param host_state Host state to update
     */
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
     * @brief Create blades from input configuration
     * @param blade_inputs Blade input configurations
     * @param model Model to which the blades will be added
     * @return Vector of blades
     */
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

    /**
     * @brief Validate turbine input parameters for physical consistency
     *
     * Checks that all input parameters are within physically reasonable ranges
     * and that required parameters are properly specified. Throws exceptions
     * for invalid configurations that would lead to simulation errors.
     *
     * @param input Turbine configuration to validate
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
     * @brief Position all turbine nodes in the global coordinate system based on the
     * turbine input configuration
     *
     * This method performs a complex sequence of geometric transformations to position
     * all turbine components (tower, blades, hub, drivetrain) in their final locations
     * to achieve the correct turbine geometry.
     *
     * --------------------------------------------------------------------------
     * Transformation/assembly sequence
     * --------------------------------------------------------------------------
     *   - Tower: Rotated to align with global Z-axis (vertical)
     *   - Blades: Aligned with Z-axis -> pitched -> translated to hub radius -> coned ->
     *     azimuthally positioned
     *   - Drivetrain nodes: Created at intermediate shaft positions
     *   - All rotor components: Tilted by shaft angle and translated to final hub position
     *
     * @param input Turbine configuration containing geometric parameters
     * @param model Structural model to add positioned nodes to
     */
    void PositionNodes(const TurbineInput& input, Model& model) {
        //--------------------------------------------------------------------------
        // Preliminary calculations
        //--------------------------------------------------------------------------

        // Define origin point for rotations
        const Array_3 origin{0., 0., 0.};

        // Calculate shaft length from tower top to hub and overhang (adjusted for shaft tilt)
        const auto shaft_length = input.tower_axis_to_rotor_apex / cos(input.shaft_tilt_angle);

        // Calculate angle between the blades
        const double blade_angle_delta = 2. * M_PI / static_cast<double>(this->blades.size());

        // Calculate rotation quaternion to align a component from x-axis to z-axis
        const auto q_x_to_z = RotationVectorToQuaternion({0., -M_PI / 2., 0.});

        // Calculate pitch rotation quaternion (around x-axis)
        const auto q_pitch = RotationVectorToQuaternion({0., 0., input.blade_pitch_angle});

        // Calculate cone angle rotation quaternion (around y-axis)
        const auto q_cone = RotationVectorToQuaternion({0., -input.cone_angle, 0.});

        //--------------------------------------------------------------------------
        // Position tower
        //--------------------------------------------------------------------------

        // Rotate tower to align with global Z-axis
        model.RotateBeamAboutPoint(this->tower.beam_element_id, q_x_to_z, origin);

        // Calculate hub position based on tower top and rotor apex offset
        const auto& tower_top_node = model.GetNode(this->tower.nodes.back().id);
        const Array_3 apex_position{
            tower_top_node.x0[0] - input.tower_axis_to_rotor_apex,  // horizontal offset
            tower_top_node.x0[1],
            tower_top_node.x0[2] + input.tower_top_to_rotor_apex,  // vertical offset
        };

        //--------------------------------------------------------------------------
        // Position drivetrain
        //--------------------------------------------------------------------------

        // Create hub node at origin and translate to hub position based on rotor apex and hub CM
        // offset
        this->hub_node = NodeData(
            model.AddNode().SetPosition({-input.rotor_apex_to_hub, 0., 0., 1., 0., 0., 0.}).Build()
        );

        // Create azimuth node at shaft base node
        this->azimuth_node =
            NodeData(model.AddNode().SetPosition({shaft_length, 0., 0., 1., 0., 0., 0.}).Build());

        // Create shaft base node relative to hub
        this->shaft_base_node =
            NodeData(model.AddNode().SetPosition({shaft_length, 0., 0., 1., 0., 0., 0.}).Build());

        // Create vector of rotor node IDs (hub, azimuth, shaft base)
        std::vector<size_t> rotor_node_ids{
            this->hub_node.id, this->azimuth_node.id, this->shaft_base_node.id
        };

        // Create yaw bearing node at tower top position
        this->yaw_bearing_node = NodeData(
            model.AddNode().SetPosition(tower_top_node.x0).SetOrientation({1., 0., 0., 0.}).Build()
        );

        //--------------------------------------------------------------------------
        // Position blades
        //--------------------------------------------------------------------------

        // Loop over blades
        for (auto i = 0U; i < this->blades.size(); ++i) {
            // Get node IDs for this blade
            std::vector<size_t> blade_node_ids;
            std::transform(
                this->blades[i].nodes.begin(), this->blades[i].nodes.end(),
                std::back_inserter(blade_node_ids),
                [](const NodeData& node) {
                    return node.id;
                }
            );

            //----------------------------------------------------
            // Blade alignment and pitch transformation
            //----------------------------------------------------

            // Loop through node IDs and rotate them to align with global Z-axis,
            // apply pitch rotation, then translate to hub radius so blade root
            // is at the hub radius
            for (const auto& node_id : blade_node_ids) {
                model.GetNode(node_id)
                    .RotateAboutPoint(q_x_to_z, origin)  // Rotate to align with global Z-axis
                    .RotateAboutPoint(q_pitch, origin)   // Apply initial blade pitch rotation
                    .Translate({0., 0., input.hub_diameter / 2.});  // Translate to hub radius
            }

            //----------------------------------------------------
            // Blade apex node creation
            //----------------------------------------------------

            // Add blade apex node at the origin and add it to blade node IDs vector
            const auto apex_node_id =
                model.AddNode().SetPosition({0., 0., 0., 1., 0., 0., 0.}).Build();
            blade_node_ids.push_back(apex_node_id);

            // Add apex node to apex nodes vector
            this->apex_nodes.emplace_back(apex_node_id);

            //----------------------------------------------------
            // Blade cone and azimuth transformations
            //----------------------------------------------------

            // Calculate azimuth angle rotation quaternion
            const auto q_azimuth =
                RotationVectorToQuaternion({static_cast<double>(i) * blade_angle_delta, 0., 0.});

            // Rotate blade nodes (including apex node) to cone angle and then to azimuth angle
            for (const auto& node_id : blade_node_ids) {
                model.GetNode(node_id)
                    .RotateAboutPoint(q_cone, origin)      // Rotate to cone angle
                    .RotateAboutPoint(q_azimuth, origin);  // Rotate to azimuth angle
            }

            // Add blade (including apex node) IDs to rotor node IDs
            rotor_node_ids.insert(
                rotor_node_ids.end(), blade_node_ids.begin(), blade_node_ids.end()
            );
        }

        //--------------------------------------------------------------------------
        // Position rotor i.e. final rotor assembly
        //--------------------------------------------------------------------------

        // Create shaft rotation quaternion
        const auto q_shaft_tilt = RotationVectorToQuaternion({0., input.shaft_tilt_angle, 0.});

        // Rotate rotor nodes by shaft tilt angle and translate to apex position
        // At this point, rotor nodes contain:
        // hub node + azimuth node + shaft base node + blade nodes + apex nodes
        for (const auto& node_id : rotor_node_ids) {
            model.GetNode(node_id)
                .RotateAboutPoint(q_shaft_tilt, origin)  // Rotate about shaft tilt
                .Translate(apex_position);               // Translate to apex position
        }
    }

    /**
     * @brief Creates all kinematic constraints and control connections for the turbine
     *
     * This method establishes the complete constraint system that defines the turbine's
     * degrees of freedom and control interfaces. Constraints are added in a specific
     * order to build the kinematic chain from the fixed tower base to the controllable
     * rotor and blade systems.
     *
     * @param input Turbine configuration containing constraint parameters
     * @param model Structural model to add constraints to
     *
     * @note Constraints must be added after all nodes are positioned since they
     *       depend on the initial node positions and orientations for proper setup
     */
    void AddConstraints(const TurbineInput& input, Model& model) {
        // Loop through blades
        for (auto i = 0U; i < this->blades.size(); ++i) {
            // Get the blade apex node
            const auto& apex_node = model.GetNode(this->apex_nodes[i].id);

            // Get the blade root node
            const auto& root_node =
                model.GetNode(this->blades[i].nodes[0].id);  // first node of blade

            // Calculate the pitch axis for the blade (from apex to root)
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
     * @brief Set initial displacements and velocities of the nodes
     * @param input Turbine configuration
     * @param model Structural model
     */
    void SetInitialConditions(const TurbineInput& input, Model& model) {
        SetInitialRotorVelocity(input, model);
    }

    /**
     * @brief Set initial rotational velocity about the shaft axis for rotor components
     *
     * @details This method applies initial rotational velocity to all rotor components
     * including the hub, azimuth node, blade nodes, and apex nodes. The velocity is
     * calculated based on the specified rotor speed and shaft tilt angle.
     *
     * @param input Turbine configuration containing rotor speed
     * @param model Structural model
     */
    void SetInitialRotorVelocity(const TurbineInput& input, Model& model) {
        // Calculate shaft axis in global coordinates (after shaft tilt)
        const Array_3 shaft_axis{-cos(input.shaft_tilt_angle), 0., sin(input.shaft_tilt_angle)};

        // Get hub position as rotation center
        const auto& hub = model.GetNode(this->hub_node.id);
        const Array_3 rotation_center{hub.x0[0], hub.x0[1], hub.x0[2]};

        // Collect all rotor node IDs (hub, azimuth, blade nodes, and apex nodes)
        std::vector<size_t> rotor_node_ids{this->hub_node.id, this->azimuth_node.id};

        // Add all blade nodes and apex nodes
        for (size_t i = 0; i < this->blades.size(); ++i) {
            // Add blade nodes
            for (const auto& blade_node : this->blades[i].nodes) {
                rotor_node_ids.push_back(blade_node.id);
            }
            // Add apex node
            rotor_node_ids.push_back(this->apex_nodes[i].id);
        }

        // Create rigid body velocity -> transl. vel = 0, angular vel. about shaft axis
        const Array_6 rigid_body_velocity{
            0.,
            0.,
            0.,
            input.rotor_speed * shaft_axis[0],
            input.rotor_speed * shaft_axis[1],
            input.rotor_speed * shaft_axis[2]
        };

        // Apply rotational velocity to all rotor nodes
        for (const auto& node_id : rotor_node_ids) {
            model.GetNode(node_id).SetVelocityAboutPoint(rigid_body_velocity, rotation_center);
        }
    }
};

}  // namespace openturbine::interfaces::components
