#pragma once

#include <functional>

#include "node.hpp"

#include "src/restruct_poc/beams/beam_element.hpp"
#include "src/restruct_poc/solver/constraint.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

// InvalidNode represents an invalid node in constraints that only use the target node.
static const Node InvalidNode(0U, {0., 0., 0., 1., 0., 0., 0.});

/// @brief Struct to define a turbine model with nodes and constraints
/// @details A model is a collection of nodes and constraints that define the geometry and
/// relationships between components in a turbine. Nodes represent points in space with
/// position, displacement, velocity, and acceleration. Constraints represent relationships
/// between nodes that restrict their relative motion in some way.
class Model {
public:
    Model() = default;

    Model(
        const std::vector<Node>& nodes, const std::vector<BeamElement>& beam_elements,
        const std::vector<Constraint>& constraints
    ) {
        for (const auto& n : nodes) {
            this->nodes_.emplace_back(std::make_shared<Node>(n));
        }
        for (const auto& e : beam_elements) {
            this->beam_elements_.emplace_back(std::make_shared<BeamElement>(e));
        }
        for (const auto& c : constraints) {
            this->constraints_.emplace_back(std::make_shared<Constraint>(c));
        }
    }

    Model(
        std::vector<std::shared_ptr<Node>> nodes,
        std::vector<std::shared_ptr<BeamElement>> beam_elements,
        std::vector<std::shared_ptr<Constraint>> constraints
    )
        : nodes_(std::move(nodes)),
          beam_elements_(std::move(beam_elements)),
          constraints_(std::move(constraints)) {}

    /// Add a node to the model and return a shared pointer to the node
    std::shared_ptr<Node> AddNode(
        const Array_7& position, const Array_7& displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        const Array_6& velocity = Array_6{0., 0., 0., 0., 0., 0.},
        const Array_6& acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    ) {
        return this->nodes_.emplace_back(
            std::make_shared<Node>(nodes_.size(), position, displacement, velocity, acceleration)
        );
    }

    /// Return a node by ID - const/read-only version
    [[nodiscard]] const Node& GetNode(size_t id) const { return *this->nodes_[id]; }

    /// Return a node by ID - non-const version
    [[nodiscard]] Node& GetNode(size_t id) { return *this->nodes_[id]; }

    /// Returns a reference to the nodes in the model (as vector of shared pointers)
    [[nodiscard]] const std::vector<std::shared_ptr<Node>>& GetNodes() const { return this->nodes_; }

    /// Returns the number of nodes in the model
    [[nodiscard]] size_t NumNodes() const { return this->nodes_.size(); }

    /// Add a beam element to the model and return a shared pointer to the element
    std::shared_ptr<BeamElement> AddBeamElement(
        const std::vector<BeamNode>& nodes, const std::vector<BeamSection>& sections,
        const BeamQuadrature& quadrature
    ) {
        return this->beam_elements_.emplace_back(
            std::make_shared<BeamElement>(nodes, sections, quadrature)
        );
    }

    /// Return a beam element by ID - const/read-only version
    [[nodiscard]] const BeamElement& GetBeamElement(size_t id) const {
        return *this->beam_elements_[id];
    }

    /// Return a beam element by ID - non-const version
    [[nodiscard]] BeamElement& GetBeamElement(size_t id) { return *this->beam_elements_[id]; }

    /// Returns a reference to the beam elements in the model
    [[nodiscard]] const std::vector<std::shared_ptr<BeamElement>>& GetBeamElements() const {
        return this->beam_elements_;
    }

    /// Returns the number of beam elements in the model
    [[nodiscard]] size_t NumBeamElements() const { return this->beam_elements_.size(); }

    /// Adds a rigid constraint to the model and returns the constraint
    std::shared_ptr<Constraint> AddRigidJointConstraint(const Node& node1, const Node& node2) {
        return this->constraints_.emplace_back(std::make_shared<Constraint>(
            ConstraintType::kRigidJoint, constraints_.size(), node1, node2
        ));
    }

    /// Adds a prescribed boundary condition constraint to the model and returns the constraint
    std::shared_ptr<Constraint> AddPrescribedBC(
        const Node& node, const Array_3& ref_position = {0., 0., 0.}
    ) {
        return this->constraints_.emplace_back(std::make_shared<Constraint>(
            ConstraintType::kPrescribedBC, constraints_.size(), InvalidNode, node, ref_position
        ));
    }

    /// Adds a fixed boundary condition constraint to the model and returns the constraint
    std::shared_ptr<Constraint> AddFixedBC(const Node& node) {
        return this->constraints_.emplace_back(std::make_shared<Constraint>(
            ConstraintType::kFixedBC, constraints_.size(), InvalidNode, node
        ));
    }

    /// Adds a revolute/hinge constraint to the model and returns the constraint
    std::shared_ptr<Constraint> AddRevoluteJointConstraint(
        const Node& node1, const Node& node2, const Array_3& axis, double* torque
    ) {
        return this->constraints_.emplace_back(std::make_shared<Constraint>(
            ConstraintType::kRevoluteJoint, constraints_.size(), node1, node2, axis, torque
        ));
    }

    /// Adds a rotation control constraint to the model and returns the constraint
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

    /// Returns the number of constraints in the model
    [[nodiscard]] size_t NumConstraints() const { return this->constraints_.size(); }

private:
    std::vector<std::shared_ptr<Node>> nodes_;                 //< Nodes in the model
    std::vector<std::shared_ptr<BeamElement>> beam_elements_;  //< Beam elements in the model
    std::vector<std::shared_ptr<Constraint>> constraints_;     //< Constraints in the model
};

}  // namespace openturbine