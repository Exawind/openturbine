#pragma once

#include <functional>

#include "copy_nodes_to_state.hpp"
#include "node.hpp"

#include "src/constraints/constraint.hpp"
#include "src/elements/beams/beam_element.hpp"
#include "src/elements/masses/mass_element.hpp"
#include "src/state/state.hpp"
#include "src/types.hpp"

namespace openturbine {

/// Represents an invalid node in constraints that only uses the target node
static const Node InvalidNode(0U, {0., 0., 0., 1., 0., 0., 0.});

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

    /**
     * @brief Constructs a model from vectors of nodes, beam elements, and constraints
     *
     * @param nodes Vector of nodes
     * @param beam_elements Vector of beam elements
     * @param mass_elements Vector of mass elements
     * @param constraints Vector of constraints
     */
    Model(
        const std::vector<Node>& nodes, const std::vector<BeamElement>& beam_elements,
        const std::vector<MassElement>& mass_elements, const std::vector<Constraint>& constraints
    ) {
        for (const auto& n : nodes) {
            this->nodes_.emplace_back(std::make_shared<Node>(n));
        }
        for (const auto& e : beam_elements) {
            this->beam_elements_.emplace_back(std::make_shared<BeamElement>(e));
        }
        for (const auto& m : mass_elements) {
            this->mass_elements_.emplace_back(std::make_shared<MassElement>(m));
        }
        for (const auto& c : constraints) {
            this->constraints_.emplace_back(std::make_shared<Constraint>(c));
        }
    }

    /**
     * @brief Constructs a model from vectors of shared pointers to nodes, beam elements, and
     * constraints
     *
     * @param nodes Vector of shared pointers to nodes
     * @param beam_elements Vector of shared pointers to beam elements
     * @param mass_elements Vector of shared pointers to mass elements
     * @param constraints Vector of shared pointers to constraints
     */
    Model(
        std::vector<std::shared_ptr<Node>> nodes,
        std::vector<std::shared_ptr<BeamElement>> beam_elements,
        std::vector<std::shared_ptr<MassElement>> mass_elements,
        std::vector<std::shared_ptr<Constraint>> constraints
    )
        : nodes_(std::move(nodes)),
          beam_elements_(std::move(beam_elements)),
          mass_elements_(std::move(mass_elements)),
          constraints_(std::move(constraints)) {}

    /**
     * @brief Adds a node to the model and returns a shared pointer to the node
     *
     * @param position Position vector
     * @param displacement Displacement vector
     * @param velocity Velocity vector
     * @param acceleration Acceleration vector
     */
    std::shared_ptr<Node> AddNode(
        const Array_7& position, const Array_7& displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        const Array_6& velocity = Array_6{0., 0., 0., 0., 0., 0.},
        const Array_6& acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    ) {
        return this->nodes_.emplace_back(
            std::make_shared<Node>(nodes_.size(), position, displacement, velocity, acceleration)
        );
    }

    /// Returns a node by ID - const/read-only version
    [[nodiscard]] const Node& GetNode(size_t id) const { return *this->nodes_[id]; }

    /// Returns a node by ID - non-const version
    [[nodiscard]] Node& GetNode(size_t id) { return *this->nodes_[id]; }

    /// Returns a reference to the nodes in the model (as vector of shared pointers)
    [[nodiscard]] const std::vector<std::shared_ptr<Node>>& GetNodes() const { return this->nodes_; }

    /// Returns the number of nodes present in the model
    [[nodiscard]] size_t NumNodes() const { return this->nodes_.size(); }

    /// Adds a beam element to the model and returns a shared pointer handle to the element
    std::shared_ptr<BeamElement> AddBeamElement(
        const std::vector<BeamNode>& nodes, const std::vector<BeamSection>& sections,
        const BeamQuadrature& quadrature
    ) {
        return this->beam_elements_.emplace_back(
            std::make_shared<BeamElement>(nodes, sections, quadrature)
        );
    }

    /// Returns a beam element by ID - const/read-only version
    [[nodiscard]] const BeamElement& GetBeamElement(size_t id) const {
        return *this->beam_elements_[id];
    }

    /// Returns a beam element by ID - non-const version
    [[nodiscard]] BeamElement& GetBeamElement(size_t id) { return *this->beam_elements_[id]; }

    /// Returns a reference to the beam elements present in the model
    [[nodiscard]] const std::vector<std::shared_ptr<BeamElement>>& GetBeamElements() const {
        return this->beam_elements_;
    }

    /// Returns the number of beam elements present in the model
    [[nodiscard]] size_t NumBeamElements() const { return this->beam_elements_.size(); }

    /// Adds a mass element to the model and returns a shared pointer to the element
    std::shared_ptr<MassElement> AddMassElement(
        const Node& node, const std::array<std::array<double, 6>, 6>& mass
    ) {
        return this->mass_elements_.emplace_back(std::make_shared<MassElement>(node, mass));
    }

    /// Adds a mass element to the model and returns a shared pointer to the element
    /// Returns a mass element by ID - const/read-only version
    [[nodiscard]] const MassElement& GetMassElement(size_t id) const {
        return *this->mass_elements_[id];
    }

    /// Returns a mass element by ID - non-const version
    [[nodiscard]] MassElement& GetMassElement(size_t id) { return *this->mass_elements_[id]; }

    /// Returns a reference to the mass elements present in the model
    [[nodiscard]] const std::vector<std::shared_ptr<MassElement>>& GetMassElements() const {
        return this->mass_elements_;
    }

    /// Returns the number of mass elements present in the model
    [[nodiscard]] size_t NumMassElements() const { return this->mass_elements_.size(); }

    /// Adds a rigid constraint to the model and returns a handle to the constraint
    std::shared_ptr<Constraint> AddRigidJointConstraint(const Node& node1, const Node& node2) {
        return this->constraints_.emplace_back(std::make_shared<Constraint>(
            ConstraintType::kRigidJoint, constraints_.size(), node1, node2
        ));
    }

    /// Adds a prescribed boundary condition constraint to the model and returns a handle to the
    /// constraint
    std::shared_ptr<Constraint> AddPrescribedBC(
        const Node& node, const Array_3& ref_position = {0., 0., 0.}
    ) {
        return this->constraints_.emplace_back(std::make_shared<Constraint>(
            ConstraintType::kPrescribedBC, constraints_.size(), InvalidNode, node, ref_position
        ));
    }

    /// Adds a fixed boundary condition constraint to the model and returns a handle to the
    /// constraint
    std::shared_ptr<Constraint> AddFixedBC(const Node& node) {
        return this->constraints_.emplace_back(std::make_shared<Constraint>(
            ConstraintType::kFixedBC, constraints_.size(), InvalidNode, node
        ));
    }

    /// Adds a revolute/hinge constraint to the model and returns a handle to the constraint
    std::shared_ptr<Constraint> AddRevoluteJointConstraint(
        const Node& node1, const Node& node2, const Array_3& axis, double* torque
    ) {
        return this->constraints_.emplace_back(std::make_shared<Constraint>(
            ConstraintType::kRevoluteJoint, constraints_.size(), node1, node2, axis, torque
        ));
    }

    /// Adds a rotation control constraint to the model and returns a handle to the constraint
    std::shared_ptr<Constraint> AddRotationControl(
        const Node& node1, const Node& node2, const Array_3& axis, double* control
    ) {
        return this->constraints_.emplace_back(std::make_shared<Constraint>(
            ConstraintType::kRotationControl, constraints_.size(), node1, node2, axis, control
        ));
    }

    /// Returns the constraints in the model (as vector of shared pointers)
    [[nodiscard]] const std::vector<std::shared_ptr<Constraint>>& GetConstraints() const {
        return this->constraints_;
    }

    /// Returns the number of constraints present in the model
    [[nodiscard]] size_t NumConstraints() const { return this->constraints_.size(); }

    /// Returns a State object initialized from the model nodes
    [[nodiscard]] State CreateState() const {
        auto state = State(this->nodes_.size());
        CopyNodesToState(state, this->nodes_);
        return state;
    }

private:
    std::vector<std::shared_ptr<Node>> nodes_;                 //< Nodes in the model
    std::vector<std::shared_ptr<BeamElement>> beam_elements_;  //< Beam elements in the model
    std::vector<std::shared_ptr<MassElement>> mass_elements_;  //< Mass elements in the model
    std::vector<std::shared_ptr<Constraint>> constraints_;     //< Constraints in the model
};

}  // namespace openturbine
