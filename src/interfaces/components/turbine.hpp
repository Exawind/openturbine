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
 * The turbine assembly consists of multiple inter-connected nodes that
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
 *     ○ <- Tower node 1 (Tower base node - fixed constraint)
 * -------------
 *  / / / /  <-  Ground / Foundation
 *
 * Nacelle/Drivetrain Nodes
 *  - Yaw bearing node: Located at tower top, allows nacelle yaw rotation
 *  - Shaft base node: Base of the main shaft within nacelle
 *  - Azimuth node: Intermediate node for rotor azimuth positioning
 *  - Hub node: Center of mass of the rotating hub assembly
 *
 *             Yaw bearing      Shaft base      Azimuth          Hub
 *                node            node           node            node
 *                 ● -------------- ● ------------ ● ------------ ● -------------
 *    yaw control  |     rigid          torque          rigid          rigid
 *     rotation    |   connection       control       connection     connection
 *   about Z-axis  |                  via revolute                    to blades
 *                                      joint
 * Blade Assembly Nodes
 *  - Blade apex nodes: Connection points between hub and blade roots (one per blade)
 *  - Blade structural nodes: Beam nodes along each blade span
 *
 *                   pitch axis,
 *                rotation control |<----- Blade nodes ----->|
 *     ● Blade apex -------------  ●  --------  ●  --------  ●
 *       node                    root                     tip
 *        │                      node                     node
 *        │ rigid connection
 *        │
 *        ● Hub node
 * --------------------------------------------------------------------------
 * Kinematic chain
 * --------------------------------------------------------------------------
 * The nodes are connected in a kinematic chain that represents the turbine's
 * degrees of freedom.
 *
 * - tower base node: fixed boundary condition
 * - tower top node <-> yaw bearing node: Yaw rotation control
 * - yaw bearing node <-> shaft base node: rigid joint
 * - shaft base node <-> azimuth node: revolute joint with torque control
 * - hub <-> blade apex nodes: rigid joint
 * - blade apex nodes <-> blade root nodes: Pitch rotation control
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

        // Add mass elements
        AddMassElements(input, model);

        // Constraints must be added after all nodes are positioned because
        // they depend on the initial positions of the nodes
        AddConstraints(input, model);

        // Set initial conditions (velocities, accelerations etc.) after positioning
        // and adding constraints
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

        // Calculate rotation quaternion to align a component from x-axis to z-axis (i.e.
        // rotate about y-axis by -90 degrees)
        const auto q_x_to_z = RotationVectorToQuaternion({0., -M_PI / 2., 0.});

        // Calculate pitch rotation quaternion (rotate about z-axis by pitch angle)
        const auto q_pitch = RotationVectorToQuaternion({0., 0., input.blade_pitch_angle});

        // Calculate cone angle rotation quaternion (rotate about y-axis by -cone angle)
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

        // Create vector of nacelle node IDs (yaw bearing, shaft base, azimuth, hub)
        std::vector<size_t> nacelle_node_ids{
            this->yaw_bearing_node.id, this->shaft_base_node.id, this->azimuth_node.id,
            this->hub_node.id
        };

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
                    .RotateAboutPoint(q_x_to_z, origin)  // Rotate to align with Z-axis
                    .RotateAboutPoint(q_pitch, origin)   // Apply initial blade pitch (around Z-axis)
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

            // Calculate azimuth angle rotation quaternion (rotate about x-axis)
            const auto q_azimuth =
                RotationVectorToQuaternion({static_cast<double>(i) * blade_angle_delta, 0., 0.});

            // Rotate blade nodes (including apex node) to cone angle and then to azimuth angle
            for (const auto& node_id : blade_node_ids) {
                model.GetNode(node_id)
                    .RotateAboutPoint(q_cone, origin)      // Rotate to cone angle (about y-axis)
                    .RotateAboutPoint(q_azimuth, origin);  // Rotate to azimuth angle (about x-axis)
            }

            // Add blade (including apex node) IDs to rotor node IDs and nacelle node IDs
            rotor_node_ids.insert(
                rotor_node_ids.end(), blade_node_ids.begin(), blade_node_ids.end()
            );
            nacelle_node_ids.insert(
                nacelle_node_ids.end(), blade_node_ids.begin(), blade_node_ids.end()
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

        //--------------------------------------------------------------------------
        // Apply initial nacelle yaw rotation
        //--------------------------------------------------------------------------

        // Apply initial yaw rotation if non-zero
        if (std::abs(input.nacelle_yaw_angle) > 1e-12) {
            // Create yaw rotation quaternion (rotation about tower Z-axis)
            const auto q_yaw = RotationVectorToQuaternion({0., 0., input.nacelle_yaw_angle});

            // Rotate all nacelle components about tower top position
            for (const auto& node_id : nacelle_node_ids) {
                model.GetNode(node_id).RotateAboutPoint(
                    q_yaw, {tower_top_node.x0[0], tower_top_node.x0[1], tower_top_node.x0[2]}
                );
            }
        }

        //--------------------------------------------------------------------------
        // Apply tower base displacement
        //--------------------------------------------------------------------------

        const auto& tower_base_node = model.GetNode(this->tower.nodes.front().id);
        const Array_3 original_tower_base_position{
            tower_base_node.x0[0], tower_base_node.x0[1], tower_base_node.x0[2]
        };

        // Calculate translation from original tower base to input position
        const Array_3 tower_base_displacement{
            input.tower_base_position[0] - original_tower_base_position[0],
            input.tower_base_position[1] - original_tower_base_position[1],
            input.tower_base_position[2] - original_tower_base_position[2]
        };
        // Get tower base orientation
        const Array_4 tower_base_orientation{
            input.tower_base_position[3], input.tower_base_position[4], input.tower_base_position[5],
            input.tower_base_position[6]
        };

        // Check if displacement is non-zero
        const auto has_displacement = std::sqrt(
                                          tower_base_displacement[0] * tower_base_displacement[0] +
                                          tower_base_displacement[1] * tower_base_displacement[1] +
                                          tower_base_displacement[2] * tower_base_displacement[2]
                                      ) > 1e-12;
        // Check if rotation is non-identity (not [1, 0, 0, 0])
        const auto has_rotation = std::abs(tower_base_orientation[0] - 1.) > 1e-12 ||
                                  std::abs(tower_base_orientation[1]) > 1e-12 ||
                                  std::abs(tower_base_orientation[2]) > 1e-12 ||
                                  std::abs(tower_base_orientation[3]) > 1e-12;
        // Apply tower base displacement if non-zero
        if (has_displacement || has_rotation) {
            // Collect all turbine node IDs (tower + nacelle + rotor + blades)
            std::vector<size_t> all_turbine_node_ids;
            all_turbine_node_ids.reserve(this->tower.nodes.size() + nacelle_node_ids.size());
            // Add tower nodes
            std::transform(
                this->tower.nodes.begin(), this->tower.nodes.end(),
                std::back_inserter(all_turbine_node_ids),
                [](const auto& tower_node) {
                    return tower_node.id;
                }
            );
            // Add nacelle nodes (already includes rotor and blade nodes)
            all_turbine_node_ids.insert(
                all_turbine_node_ids.end(), nacelle_node_ids.begin(), nacelle_node_ids.end()
            );

            // Apply displacement to all turbine nodes
            for (const auto& node_id : all_turbine_node_ids) {
                // first rotate about original tower base position
                model.GetNode(node_id).RotateAboutPoint(
                    tower_base_orientation, original_tower_base_position
                );
                // then translate to new tower base position
                model.GetNode(node_id).Translate(tower_base_displacement);
            }
        }
    }

    /**
     * @brief Adds mass elements to the turbine model at the yaw bearing and hub nodes
     *
     * @param input Turbine configuration containing inertia matrices
     * @param model Structural model to add mass elements to
     */
    /**
     * @brief Adds mass elements to the turbine model at the yaw bearing and hub nodes
     *
     * @details Creates lumped mass elements that represent:
     *  - Yaw bearing node: Combined nacelle system mass and yaw bearing mass with
     *    inertia tensor about tower-top
     *  - Hub node: Hub assembly mass and inertia properties
     *
     * @param input Turbine configuration containing inertia matrices
     * @param model Structural model to add mass elements to
     */
    void AddMassElements(const TurbineInput& input, Model& model) {
        // Add mass element at yaw bearing node
        this->yaw_bearing_mass_element_id =
            model.AddMassElement(this->yaw_bearing_node.id, input.yaw_bearing_inertia_matrix);

        // Add mass element at hub node
        this->hub_mass_element_id =
            model.AddMassElement(this->hub_node.id, input.hub_inertia_matrix);
    }

    /**
     * @brief Creates all kinematic constraints and control connections for the turbine
     *
     * This method establishes the complete constraint system that defines the turbine's
     * degrees of freedom and control interfaces. Constraints are added in a specific
     * order to build the kinematic chain from the fixed tower base to the controllable
     * rotor and blade systems.
     *
     * --------------------------------------------------------------------------
     * Constraint schematic
     * --------------------------------------------------------------------------
     *
     *           Blade 1                 Blade 2                 Blade 3
     *     ●────────●────────●     ●────────●────────●     ●────────●────────●
     *   root      mid      tip   root      mid      tip   root      mid      tip
     *    │                       │                       │
     *    │ pitch control         │ pitch control         │ pitch control
     *    │ (rotation about       │ (rotation about       │ (rotation about
     *    │  pitch axis)          │  pitch axis)          │  pitch axis)
     *    │                       │                       │
     *    ● Apex 1                ● Apex 2                ● Apex 3
     *    │                       │                       │
     *    │ rigid connection      │ rigid connection      │ rigid connection
     *    └───────────────────────┼───────────────────────┘
     *                            │
     *                            ● Hub node
     *                            │
     *                            │ rigid connection
     *                            │
     *                            ● Azimuth node
     *                            │
     *                            │ revolute joint with
     *                            │ torque control
     *                            │ (shaft rotation)
     *                            │
     *                            ● Shaft base node
     *                            │
     *                            │ rigid connection
     *                            │
     *                            ● Yaw bearing node
     *                            │
     *                            │  yaw rotation control
     *                            │ (rotation about Z-axis)
     *                            │
     *                            ● Tower top node
     *                            │
     *                            │
     *                            ● Tower node n-1
     *                            │
     *                            ● Tower node n-2
     *                            .
     *                            .
     *                            │
     *                            ● Tower node 2
     *                            │
     *                            ● Tower node 1 (base)
     *                            │
     *                            │ fixed boundary condition
     *                            │  (all DOFs constrained)
     *                        ────┴────
     *                       / / / / / /  <- Foundation
     *
     *
     * Constraint Types:
     * - Fixed bc: Constrains all 6 DOFs (tower base to foundation)
     * - Rigid joint: Constrains relative motion between two nodes (6 DOFs)
     * - Revolute joint: Allows rotation about one axis, constrains other 5 DOFs
     * - Rotation control: Controlled rotation about specified axis
     *
     * Control Interfaces:
     * - Yaw control: Controls nacelle orientation (yaw bearing rotation)
     * - Torque control: Controls rotor speed (shaft base to azimuth rotation)
     * - Pitch control: Controls blade pitch angles (apex to root rotation)
     *
     * @param input Turbine configuration containing constraint parameters
     * @param model Structural model to add constraints to
     */
    void AddConstraints(const TurbineInput& input, Model& model) {
        //--------------------------------------------------------------------------
        // Blade control constraints
        //--------------------------------------------------------------------------

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

        //--------------------------------------------------------------------------
        // Drivetrain constraints
        //--------------------------------------------------------------------------

        // Add rigid constraint between hub and azimuth node
        this->azimuth_to_hub =
            ConstraintData(model.AddRigidJointConstraint({this->azimuth_node.id, this->hub_node.id})
            );

        // Shaft axis constraint - add revolute joint between shaft base and azimuth node
        const Array_3 shaft_axis{-cos(input.shaft_tilt_angle), 0., sin(input.shaft_tilt_angle)};
        this->shaft_base_to_azimuth = ConstraintData(model.AddRevoluteJointConstraint(
            {this->shaft_base_node.id, this->azimuth_node.id}, shaft_axis, &torque_control
        ));

        // Add rigid constraint from yaw bearing to shaft base
        this->yaw_bearing_to_shaft_base = ConstraintData(
            model.AddRigidJointConstraint({this->yaw_bearing_node.id, this->shaft_base_node.id})
        );

        //--------------------------------------------------------------------------
        // Nacelle control constraints
        //--------------------------------------------------------------------------

        // Add constraint from tower top to yaw bearing
        this->yaw_control = input.nacelle_yaw_angle;
        this->tower_top_to_yaw_bearing = ConstraintData(model.AddRotationControl(
            {this->tower.nodes.back().id, this->yaw_bearing_node.id}, {0., 0., 1.},
            &this->yaw_control
        ));

        //--------------------------------------------------------------------------
        // Tower base constraint
        //--------------------------------------------------------------------------

        // Add fixed constraint at the tower base
        this->tower_base = ConstraintData(model.AddFixedBC(this->tower.nodes.front().id));
    }

    /**
     * @brief Set initial velocities, accelerations, and other dynamic initial
     * conditions of the nodes
     * @param input Turbine configuration parameters
     * @param model Structural model
     */
    void SetInitialConditions(const TurbineInput& input, Model& model) {
        // Apply initial rotor velocity about shaft axis
        SetInitialRotorVelocity(input, model);

        // Apply initial accelerations
        // SetInitialAccelerations(input, model);
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
            std::transform(
                this->blades[i].nodes.begin(), this->blades[i].nodes.end(),
                std::back_inserter(rotor_node_ids),
                [](const auto& blade_node) {
                    return blade_node.id;
                }
            );
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
