#pragma once

#include <functional>
#include <tuple>

#include "copy_nodes_to_state.hpp"
#include "node.hpp"

#include "src/constraints/constraint.hpp"
#include "src/constraints/constraints.hpp"
#include "src/dof_management/assemble_node_freedom_allocation_table.hpp"
#include "src/dof_management/compute_node_freedom_map_table.hpp"
#include "src/dof_management/create_constraint_freedom_table.hpp"
#include "src/dof_management/create_element_freedom_table.hpp"
#include "src/elements/beams/beams_input.hpp"
#include "src/elements/beams/create_beams.hpp"
#include "src/elements/elements.hpp"
#include "src/elements/masses/create_masses.hpp"
#include "src/elements/masses/masses_input.hpp"
#include "src/elements/springs/create_springs.hpp"
#include "src/elements/springs/springs_input.hpp"
#include "src/solver/solver.hpp"
#include "src/state/state.hpp"
#include "src/step/step_parameters.hpp"
#include "src/types.hpp"

namespace openturbine {

/// Represents an invalid node in constraints that only uses the target node
static const size_t InvalidNodeID(0U);

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
     * @brief Adds a node to the model and returns a shared pointer to the node
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

    /// Adds a beam element to the model and returns a shared pointer handle to the element
    size_t AddBeamElement(
        const std::vector<size_t>& node_ids, const std::vector<BeamSection>& sections,
        const BeamQuadrature& quadrature
    ) {
        const auto elem_id = this->beam_elements_.size();
        this->beam_elements_.emplace_back(elem_id, node_ids, sections, quadrature);
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
    [[nodiscard]] BeamsInput CreateBeamsInput() { return {this->beam_elements_, this->gravity_}; }

    /// Returns Beams struct initialized with beams
    [[nodiscard]] Beams CreateBeams() {
        return openturbine::CreateBeams(this->CreateBeamsInput(), this->nodes_);
    }

    //--------------------------------------------------------------------------
    // Mass Elements
    //--------------------------------------------------------------------------

    /// Adds a mass element to the model and returns a shared pointer to the element
    size_t AddMassElement(const size_t node_id, const std::array<std::array<double, 6>, 6>& mass) {
        const auto elem_id = this->mass_elements_.size();
        this->mass_elements_.emplace_back(elem_id, node_id, mass);
        return elem_id;
    }

    /// Adds a mass element to the model and returns a shared pointer to the element
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
    [[nodiscard]] Masses CreateMasses() {
        return openturbine::CreateMasses(
            MassesInput(this->mass_elements_, this->gravity_), this->nodes_
        );
    }

    //--------------------------------------------------------------------------
    // Spring Elements
    //--------------------------------------------------------------------------

    /// Adds a spring element to the model and returns a shared pointer to the element
    size_t AddSpringElement(
        const size_t node1_id, const size_t node2_id, const double stiffness,
        const double undeformed_length
    ) {
        const auto elem_id = this->spring_elements_.size();
        this->spring_elements_.emplace_back(
            elem_id, std::array{node1_id, node2_id}, stiffness, undeformed_length
        );
        return elem_id;
    }

    /// Adds a spring element to the model and returns a shared pointer to the element
    /// Returns a spring element by ID - const/read-only version
    [[nodiscard]] const SpringElement& GetSpringElement(size_t id) const {
        return this->spring_elements_[id];
    }

    /// Returns a spring element by ID - non-const version
    [[nodiscard]] SpringElement& GetSpringElement(size_t id) { return this->spring_elements_[id]; }

    /// Returns the number of spring elements present in the model
    [[nodiscard]] size_t NumSpringElements() const { return this->spring_elements_.size(); }

    /// Returns Springs struct initialized from spring elements
    [[nodiscard]] Springs CreateSprings() {
        return openturbine::CreateSprings(SpringsInput(this->spring_elements_), this->nodes_);
    }

    //--------------------------------------------------------------------------
    // Elements
    //--------------------------------------------------------------------------

    /// Returns Elements struct initialized with elements
    [[nodiscard]] Elements CreateElements() {
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

    /// Adds a rigid constraint to the model and returns a handle to the constraint
    size_t AddRigidJointConstraint(const size_t node1_id, const size_t node2_id) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            ConstraintType::kRigidJoint, constraints_.size(), node1_id, node2_id
        );
        return id;
    }

    /// Adds a prescribed boundary condition constraint to the model and returns a handle to the
    /// constraint
    size_t AddPrescribedBC(const size_t node_id, const Array_3& ref_position = {0., 0., 0.}) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            ConstraintType::kPrescribedBC, constraints_.size(), InvalidNodeID, node_id, ref_position
        );
        return id;
    }

    /// Adds a fixed boundary condition constraint to the model and returns a handle to the
    /// constraint
    size_t AddFixedBC(const size_t node_id) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            ConstraintType::kFixedBC, constraints_.size(), InvalidNodeID, node_id
        );
        return id;
    }

    /// Adds a revolute/hinge constraint to the model and returns a handle to the constraint
    size_t AddRevoluteJointConstraint(
        const size_t node1_id, const size_t node2_id, const Array_3& axis, double* torque
    ) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            ConstraintType::kRevoluteJoint, constraints_.size(), node1_id, node2_id, axis, torque
        );
        return id;
    }

    /// Adds a rotation control constraint to the model and returns a handle to the constraint
    size_t AddRotationControl(
        const size_t node1_id, const size_t node2_id, const Array_3& axis, double* control
    ) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            ConstraintType::kRotationControl, constraints_.size(), node1_id, node2_id, axis, control
        );
        return id;
    }

    /// Returns the number of constraints present in the model
    [[nodiscard]] size_t NumConstraints() const { return this->constraints_.size(); }

    /// Returns a Constraints object initialized from the model constraints
    [[nodiscard]] Constraints CreateConstraints() const {
        return Constraints(this->constraints_, this->nodes_);
    }

    //--------------------------------------------------------------------------
    // Solver
    //--------------------------------------------------------------------------

    [[nodiscard]] Solver static CreateSolver(
        State& state, Elements& elements, Constraints& constraints
    ) {
        assemble_node_freedom_allocation_table(state, elements, constraints);
        compute_node_freedom_map_table(state);
        create_element_freedom_table(elements, state);
        create_constraint_freedom_table(constraints, state);
        auto solver = Solver(
            state.ID, state.node_freedom_allocation_table, state.node_freedom_map_table,
            elements.NumberOfNodesPerElement(), elements.NodeStateIndices(), constraints.num_dofs,
            constraints.type, constraints.base_node_freedom_table,
            constraints.target_node_freedom_table, constraints.row_range
        );
        return solver;
    }

private:
    Array_3 gravity_ = {0., 0., 0.};              //< Gravity components
    std::vector<Node> nodes_;                     //< Nodes in the model
    std::vector<BeamElement> beam_elements_;      //< Beam elements in the model
    std::vector<MassElement> mass_elements_;      //< Mass elements in the model
    std::vector<SpringElement> spring_elements_;  //< Spring elements in the model
    std::vector<Constraint> constraints_;         //< Constraints in the model
};

}  // namespace openturbine
