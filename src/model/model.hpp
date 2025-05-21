#pragma once

#include <functional>
#include <tuple>

#include "constraints/constraint.hpp"
#include "constraints/constraints.hpp"
#include "copy_nodes_to_state.hpp"
#include "dof_management/assemble_node_freedom_allocation_table.hpp"
#include "dof_management/compute_node_freedom_map_table.hpp"
#include "dof_management/create_constraint_freedom_table.hpp"
#include "dof_management/create_element_freedom_table.hpp"
#include "elements/beams/beams.hpp"
#include "elements/beams/beams_input.hpp"
#include "elements/beams/calculate_QP_deformation.hpp"
#include "elements/beams/create_beams.hpp"
#include "elements/beams/interpolate_to_quadrature_points.hpp"
#include "elements/elements.hpp"
#include "elements/masses/create_masses.hpp"
#include "elements/masses/masses_input.hpp"
#include "elements/springs/create_springs.hpp"
#include "elements/springs/springs_input.hpp"
#include "mesh_connectivity.hpp"
#include "node.hpp"
#include "solver/solver.hpp"
#include "state/state.hpp"
#include "step/step_parameters.hpp"
#include "system/beams/update_node_state.hpp"
#include "types.hpp"

namespace openturbine {

/// Represents an invalid node in constraints that only uses the target node
static const size_t InvalidNodeID(0U);

/// @brief Compute freedom tables for state, elements, and constraints, then construct and return
/// solver.
[[nodiscard]] inline Solver CreateSolver(
    State& state, Elements& elements, Constraints& constraints
) {
    assemble_node_freedom_allocation_table(state, elements, constraints);
    compute_node_freedom_map_table(state);
    create_element_freedom_table(elements, state);
    create_constraint_freedom_table(constraints, state);
    return {
        state.ID,
        state.active_dofs,
        state.node_freedom_map_table,
        elements.NumberOfNodesPerElement(),
        elements.NodeStateIndices(),
        constraints.num_dofs,
        constraints.base_active_dofs,
        constraints.target_active_dofs,
        constraints.base_node_freedom_table,
        constraints.target_node_freedom_table,
        constraints.row_range,
    };
}

/**
 * @brief Struct to define a turbine model with nodes, elements, and constraints
 *
 * @details A model is a collection of nodes, elements, and constraints that define the geometry and
 * relationships between components in a turbine
 * - Nodes represent points in space with position, displacement, velocity, and acceleration
 * - Elements represent components in the turbine with mass (beams, rigid bodies) and stiffness
 * matrices (beams)
 * - Constraints represent relationships between nodes that restrict their relative motion in some
 * way
 */
class Model {
public:
    /// Default constructor
    Model() = default;

    // Constructor with gravity specified
    explicit Model(std::array<double, 3> gravity) : gravity_(gravity) {}

    //--------------------------------------------------------------------------
    // Miscellaneous
    //--------------------------------------------------------------------------

    /// @brief Sets the gravity components for the model
    /// @param x X gravity component
    /// @param y Y gravity component
    /// @param z Z gravity component
    void SetGravity(double x, double y, double z) {
        this->gravity_[0] = x;
        this->gravity_[1] = y;
        this->gravity_[2] = z;
    }

    //--------------------------------------------------------------------------
    // Nodes
    //--------------------------------------------------------------------------

    /**
     * @brief Adds a node to the model and returns the index of the node
     *
     * @param position Position vector
     * @param displacement Displacement vector
     * @param velocity Velocity vector
     * @param acceleration Acceleration vector
     */
    NodeBuilder AddNode() {
        const auto id = this->nodes_.size();
        this->nodes_.emplace_back(id);
        return NodeBuilder(this->nodes_.back());
    }

    /// Returns a node by ID - const/read-only version
    [[nodiscard]] const Node& GetNode(size_t id) const { return this->nodes_[id]; }

    /// Returns a node by ID - non-const version
    [[nodiscard]] Node& GetNode(size_t id) { return this->nodes_[id]; }

    /// Returns the number of nodes present in the model
    [[nodiscard]] size_t NumNodes() const { return this->nodes_.size(); }

    /// Returns constant reference to nodes vector
    [[nodiscard]] const std::vector<Node>& GetNodes() const { return this->nodes_; }

    //--------------------------------------------------------------------------
    // Beam Elements
    //--------------------------------------------------------------------------

    /// Adds a beam element to the model and returns the index of the element
    size_t AddBeamElement(
        const std::vector<size_t>& node_ids, const std::vector<BeamSection>& sections,
        const BeamQuadrature& quadrature
    ) {
        const auto elem_id = this->beam_elements_.size();
        this->beam_elements_.emplace_back(elem_id, node_ids, sections, quadrature);
        this->mesh_connectivity_.AddBeamElementConnectivity(elem_id, node_ids);
        return elem_id;
    }

    /// Returns a beam element by ID - const/read-only version
    [[nodiscard]] const BeamElement& GetBeamElement(size_t id) const {
        return this->beam_elements_[id];
    }

    /// Returns a beam element by ID - non-const version
    [[nodiscard]] BeamElement& GetBeamElement(size_t id) { return this->beam_elements_[id]; }

    /// Returns a reference to the beam elements present in the model
    [[nodiscard]] const std::vector<BeamElement>& GetBeamElements() const {
        return this->beam_elements_;
    }

    /// Returns the number of beam elements present in the model
    [[nodiscard]] size_t NumBeamElements() const { return this->beam_elements_.size(); }

    /// Returns initialized BeamsInput struct
    [[nodiscard]] BeamsInput CreateBeamsInput() const {
        return {this->beam_elements_, this->gravity_};
    }

    /// Returns Beams struct initialized with beams
    [[nodiscard]] Beams CreateBeams() const {
        return openturbine::CreateBeams(this->CreateBeamsInput(), this->nodes_);
    }

    /// Translate all beam nodes by given displacement
    void TranslateBeam(size_t beam_elem_id, const Array_3& displacement) {
        const auto& beam_elem = this->beam_elements_[beam_elem_id];
        for (const auto& node_id : beam_elem.node_ids) {
            this->GetNode(node_id).Translate(displacement);
        }
    }

    /// Rotate all beam nodes by given displacement quaternion about origin point
    void RotateBeamAboutPoint(
        size_t beam_elem_id, const Array_4& displacement_quaternion, const Array_3& point
    ) {
        const auto& beam_elem = this->beam_elements_[beam_elem_id];
        for (const auto& node_id : beam_elem.node_ids) {
            this->GetNode(node_id).RotateAboutPoint(displacement_quaternion, point);
        }
    }

    void SetBeamVelocityAboutPoint(
        size_t beam_elem_id, const Array_6& velocity, const Array_3& point
    ) {
        const auto& beam_elem = this->beam_elements_[beam_elem_id];
        for (const auto& node_id : beam_elem.node_ids) {
            // Get node
            auto& node = this->GetNode(node_id);

            // Get distance from reference point to node
            const Array_3 r{node.x[0] - point[0], node.x[1] - point[1], node.x[2] - point[2]};

            // Get angular velocity
            const Array_3 omega{velocity[3], velocity[4], velocity[5]};

            // Calculate velocity contribution from angular velocity
            const auto omega_cross_r = CrossProduct(omega, r);

            // Set node translation velocity
            node.v[0] = velocity[0] + omega_cross_r[0];
            node.v[1] = velocity[1] + omega_cross_r[1];
            node.v[2] = velocity[2] + omega_cross_r[2];

            // Set node angular velocity
            node.v[3] = velocity[3];
            node.v[4] = velocity[4];
            node.v[5] = velocity[5];
        }
    }

    void SetBeamAccelerationAboutPoint(
        size_t beam_elem_id, const Array_6& acceleration, const Array_3& omega, const Array_3& point
    ) {
        const auto& beam_elem = this->beam_elements_[beam_elem_id];
        for (const auto& node_id : beam_elem.node_ids) {
            // Get node
            auto& node = this->GetNode(node_id);

            // Get distance from reference point to node
            const Array_3 r{node.x[0] - point[0], node.x[1] - point[1], node.x[2] - point[2]};

            // Get angular acceleration
            const Array_3 alpha{acceleration[3], acceleration[4], acceleration[5]};

            // Calculate translational acceleration contribution from angular acceleration
            const auto alpha_cross_r = CrossProduct(alpha, r);

            // Calculate translational acceleration contribution from angular velocity
            const auto omega_cross_omega_cross_r = CrossProduct(omega, CrossProduct(omega, r));

            // Set node translation acceleration
            node.vd[0] = acceleration[0] + alpha_cross_r[0] + omega_cross_omega_cross_r[0];
            node.vd[1] = acceleration[1] + alpha_cross_r[1] + omega_cross_omega_cross_r[1];
            node.vd[2] = acceleration[2] + alpha_cross_r[2] + omega_cross_omega_cross_r[2];

            // Set node angular acceleration
            node.vd[3] = alpha[0];
            node.vd[4] = alpha[1];
            node.vd[5] = alpha[2];
        }
    }

    //--------------------------------------------------------------------------
    // Mass Elements
    //--------------------------------------------------------------------------

    /// Adds a mass element to the model and returns the index of the element
    size_t AddMassElement(const size_t node_id, const std::array<std::array<double, 6>, 6>& mass) {
        const auto elem_id = this->mass_elements_.size();
        this->mass_elements_.emplace_back(elem_id, node_id, mass);
        this->mesh_connectivity_.AddMassElementConnectivity(elem_id, node_id);
        return elem_id;
    }

    /// Returns a mass element by ID - const/read-only version
    [[nodiscard]] const MassElement& GetMassElement(size_t id) const {
        return this->mass_elements_[id];
    }

    /// Returns a mass element by ID - non-const version
    [[nodiscard]] MassElement& GetMassElement(size_t id) { return this->mass_elements_[id]; }

    /// Returns a reference to the mass elements present in the model
    [[nodiscard]] const std::vector<MassElement>& GetMassElements() const {
        return this->mass_elements_;
    }

    /// Returns the number of mass elements present in the model
    [[nodiscard]] size_t NumMassElements() const { return this->mass_elements_.size(); }

    /// Returns Masses struct initialized from mass elements
    [[nodiscard]] Masses CreateMasses() const {
        return openturbine::CreateMasses(
            MassesInput(this->mass_elements_, this->gravity_), this->nodes_
        );
    }

    //--------------------------------------------------------------------------
    // Spring Elements
    //--------------------------------------------------------------------------

    /// Adds a spring element to the model and returns the index of the element
    size_t AddSpringElement(
        const size_t node1_id, const size_t node2_id, const double stiffness,
        const double undeformed_length
    ) {
        const auto elem_id = this->spring_elements_.size();
        this->spring_elements_.emplace_back(
            elem_id, std::array{node1_id, node2_id}, stiffness, undeformed_length
        );
        this->mesh_connectivity_.AddSpringElementConnectivity(
            elem_id, std::array{node1_id, node2_id}
        );
        return elem_id;
    }

    /// Returns a spring element by ID - const/read-only version
    [[nodiscard]] const SpringElement& GetSpringElement(size_t id) const {
        return this->spring_elements_[id];
    }

    /// Returns a spring element by ID - non-const version
    [[nodiscard]] SpringElement& GetSpringElement(size_t id) { return this->spring_elements_[id]; }

    /// Returns the number of spring elements present in the model
    [[nodiscard]] size_t NumSpringElements() const { return this->spring_elements_.size(); }

    /// Returns Springs struct initialized from spring elements
    [[nodiscard]] Springs CreateSprings() const {
        return openturbine::CreateSprings(SpringsInput(this->spring_elements_), this->nodes_);
    }

    //--------------------------------------------------------------------------
    // Elements
    //--------------------------------------------------------------------------

    /// Returns Elements struct initialized with elements
    [[nodiscard]] Elements CreateElements() const {
        return Elements{
            this->CreateBeams(),
            this->CreateMasses(),
            this->CreateSprings(),
        };
    }

    //--------------------------------------------------------------------------
    // State
    //--------------------------------------------------------------------------

    /// Returns a State object initialized from the model nodes
    [[nodiscard]] State CreateState() const {
        auto state = State(this->nodes_.size());
        CopyNodesToState(state, this->nodes_);
        return state;
    }

    //--------------------------------------------------------------------------
    // Constraints
    //--------------------------------------------------------------------------

    /// Adds a fixed boundary condition constraint to the model and returns the ID
    size_t AddFixedBC(const size_t node_id) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::kFixedBC, std::array{InvalidNodeID, node_id}
        );
        this->mesh_connectivity_.AddConstraintConnectivity(id, std::vector<size_t>{node_id});
        return id;
    }

    /// Adds a prescribed boundary condition constraint to the model and returns the ID
    size_t AddPrescribedBC(const size_t node_id, const Array_3& ref_position = {0., 0., 0.}) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::kPrescribedBC, std::array{InvalidNodeID, node_id}, ref_position
        );
        this->mesh_connectivity_.AddConstraintConnectivity(id, std::vector<size_t>{node_id});
        return id;
    }

    /// Adds a rigid constraint to the model and returns the ID
    size_t AddRigidJointConstraint(const std::array<size_t, 2>& node_ids) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(id, ConstraintType::kRigidJoint, node_ids);
        this->mesh_connectivity_.AddConstraintConnectivity(
            id, std::vector<size_t>{node_ids[0], node_ids[1]}
        );
        return id;
    }

    /// Adds a revolute/hinge constraint to the model and returns the ID
    size_t AddRevoluteJointConstraint(
        const std::array<size_t, 2>& node_ids, const Array_3& axis, double* torque
    ) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(id, ConstraintType::kRevoluteJoint, node_ids, axis, torque);
        this->mesh_connectivity_.AddConstraintConnectivity(
            id, std::vector<size_t>{node_ids[0], node_ids[1]}
        );
        return id;
    }

    /// Adds a rotation control constraint to the model and returns the ID
    size_t AddRotationControl(
        const std::array<size_t, 2>& node_ids, const Array_3& axis, double* control
    ) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::kRotationControl, node_ids, axis, control
        );
        this->mesh_connectivity_.AddConstraintConnectivity(
            id, std::vector<size_t>{node_ids[0], node_ids[1]}
        );
        return id;
    }

    /// Adds a fixed boundary condition constraint (6DOFs to 3DOFs) to the model and returns the ID
    size_t AddFixedBC3DOFs(const size_t node_id) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::kFixedBC3DOFs, std::array{InvalidNodeID, node_id}
        );
        this->mesh_connectivity_.AddConstraintConnectivity(id, std::vector<size_t>{node_id});
        return id;
    }

    /// Adds a prescribed boundary condition constraint (6DOFs to 3DOFs) to the model and returns the
    /// ID
    size_t AddPrescribedBC3DOFs(const size_t node_id, const Array_3& ref_position = {0., 0., 0.}) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::kPrescribedBC3DOFs, std::array{InvalidNodeID, node_id}, ref_position
        );
        this->mesh_connectivity_.AddConstraintConnectivity(id, std::vector<size_t>{node_id});
        return id;
    }

    /// Adds a rigid joint constraint (6DOFs to 3DOFs) to the model and returns the ID
    size_t AddRigidJoint6DOFsTo3DOFs(const std::array<size_t, 2>& node_ids) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(id, ConstraintType::kRigidJoint6DOFsTo3DOFs, node_ids);
        this->mesh_connectivity_.AddConstraintConnectivity(
            id, std::vector<size_t>{node_ids[0], node_ids[1]}
        );
        return id;
    }

    /// Returns the number of constraints present in the model
    [[nodiscard]] size_t NumConstraints() const { return this->constraints_.size(); }

    /// Returns a Constraints object initialized from the model constraints
    [[nodiscard]] Constraints CreateConstraints() const {
        return Constraints(this->constraints_, this->nodes_);
    }

    // Returns a State, Elements, and Constraints object initialized from the model
    [[nodiscard]] std::tuple<State, Elements, Constraints> CreateSystem() const {
        return {this->CreateState(), this->CreateElements(), this->CreateConstraints()};
    }

    // Returns a State, Elements, Constraints, and Solver object initialized from the model
    [[nodiscard]] std::tuple<State, Elements, Constraints, Solver> CreateSystemWithSolver() const {
        auto [state, elements, constraints] = this->CreateSystem();
        auto solver = CreateSolver(state, elements, constraints);
        return {state, elements, constraints, solver};
    }

    //--------------------------------------------------------------------------
    // Mesh Connectivity
    //--------------------------------------------------------------------------

    /// @brief Get the mesh connectivity (const/read-only version)
    [[nodiscard]] const MeshConnectivity& GetMeshConnectivity() const {
        return this->mesh_connectivity_;
    }

    /// @brief Get mutable mesh connectivity (non-const version)
    [[nodiscard]] MeshConnectivity& GetMeshConnectivity() { return this->mesh_connectivity_; }

    /// @brief Export mesh connectivity to a YAML file
    void ExportMeshConnectivityToYAML(const std::string& filename = "mesh_connectivity.yaml") const {
        this->mesh_connectivity_.ExportToYAML(filename);
    }

private:
    Array_3 gravity_ = {0., 0., 0.};              //< Gravity components
    std::vector<Node> nodes_;                     //< Nodes in the model
    std::vector<BeamElement> beam_elements_;      //< Beam elements in the model
    std::vector<MassElement> mass_elements_;      //< Mass elements in the model
    std::vector<SpringElement> spring_elements_;  //< Spring elements in the model
    std::vector<Constraint> constraints_;         //< Constraints in the model
    MeshConnectivity mesh_connectivity_;  //< Mesh connectivity tracking element-node relationships
};

}  // namespace openturbine
