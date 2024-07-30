#pragma once

#include "node.hpp"

#include "src/restruct_poc/beams/beam_element.hpp"
#include "src/restruct_poc/solver/constraint.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

// InvalidNode represents an invalid node in constraints that only use the target node.
static Node InvalidNode(-1, {0., 0., 0., 1., 0., 0., 0.});

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
            this->nodes_.push_back(std::make_shared<Node>(n));
        }
        for (const auto& e : beam_elements) {
            this->beam_elements_.push_back(std::make_shared<BeamElement>(e));
        }
        for (const auto& c : constraints) {
            this->constraints_.push_back(std::make_shared<Constraint>(c));
        }
    }

    Model(
        std::vector<std::shared_ptr<Node>> nodes,
        std::vector<std::shared_ptr<BeamElement>> beam_elements
    )
        : nodes_(std::move(nodes)), beam_elements_(std::move(beam_elements)) {}

    // Make the class non-copyable and non-assignable
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /// Add a node to the model and return a shared pointer to the node
    std::shared_ptr<Node> AddNode(
        const Array_7& position, const Array_7& displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        const Array_6& velocity = Array_6{0., 0., 0., 0., 0., 0.},
        const Array_6& acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    ) {
        auto node =
            std::make_shared<Node>(nodes_.size(), position, displacement, velocity, acceleration);
        this->nodes_.push_back(std::move(node));
        return this->nodes_.back();
    }

    /// Return a node by ID - const/read-only version
    std::shared_ptr<const Node> GetNode(int id) const { return this->nodes_[id]; }

    /// Return a node by ID - non-const version
    std::shared_ptr<Node> GetNode(int id) { return this->nodes_[id]; }

    /// Returns a reference to the nodes in the model (as vector of shared pointers)
    const std::vector<std::shared_ptr<Node>>& GetNodes() const { return this->nodes_; }

    /// Returns the number of nodes in the model
    size_t NumNodes() const { return this->nodes_.size(); }

    /// Add a beam element to the model and return a shared pointer to the element
    std::shared_ptr<BeamElement> AddBeamElement(
        std::vector<BeamNode> nodes, std::vector<BeamSection> sections, BeamQuadrature quadrature
    ) {
        auto element = std::make_shared<BeamElement>(nodes, sections, quadrature);
        this->beam_elements_.push_back(std::move(element));
        return this->beam_elements_.back();
    }

    /// Return a beam element by ID - const/read-only version
    std::shared_ptr<const BeamElement> GetBeamElement(int id) const {
        return this->beam_elements_[id];
    }

    /// Return a beam element by ID - non-const version
    std::shared_ptr<BeamElement> GetBeamElement(int id) { return this->beam_elements_[id]; }

    /// Returns a reference to the beam elements in the model
    const std::vector<std::shared_ptr<BeamElement>>& GetBeamElements() const {
        return this->beam_elements_;
    }

    /// Returns the number of beam elements in the model
    size_t NumBeamElements() const { return this->beam_elements_.size(); }

    /// Adds a rigid constraint to the model and returns the constraint
    std::shared_ptr<Constraint> AddRigidConstraint(const Node& node1, const Node& node2) {
        auto constraint =
            std::make_shared<Constraint>(ConstraintType::kRigid, constraints_.size(), node1, node2);
        this->constraints_.push_back(std::move(constraint));
        return this->constraints_.back();
    }

    /// Adds a prescribed boundary condition constraint to the model and returns the constraint
    std::shared_ptr<Constraint> AddPrescribedBC(
        const Node& node, const Array_3& ref_position = {0., 0., 0.}
    ) {
        auto constraint = std::make_shared<Constraint>(
            ConstraintType::kPrescribedBC, constraints_.size(), InvalidNode, node, ref_position
        );
        this->constraints_.push_back(std::move(constraint));
        return this->constraints_.back();
    }

    /// Adds a fixed boundary condition constraint to the model and returns the constraint
    std::shared_ptr<Constraint> AddFixedBC(const Node& node) {
        auto constraint = std::make_shared<Constraint>(
            ConstraintType::kFixedBC, constraints_.size(), InvalidNode, node
        );
        this->constraints_.push_back(std::move(constraint));
        return this->constraints_.back();
    }

    /// Adds a cylindrical constraint to the model and returns the constraint
    std::shared_ptr<Constraint> AddCylindricalConstraint(const Node& node1, const Node& node2) {
        auto constraint = std::make_shared<Constraint>(
            ConstraintType::kCylindrical, constraints_.size(), node1, node2
        );
        this->constraints_.push_back(std::move(constraint));
        return this->constraints_.back();
    }

    /// Adds a rotation control constraint to the model and returns the constraint
    std::shared_ptr<Constraint> AddRotationControl(
        const Node& node1, const Node& node2, const Array_3& axis, float* control
    ) {
        auto constraint = std::make_shared<Constraint>(
            ConstraintType::kRotationControl, constraints_.size(), node1, node2, axis, control
        );
        this->constraints_.push_back(std::move(constraint));
        return this->constraints_.back();
    }

    /// Returns the constraints in the model (as vector of shared pointers)
    const std::vector<std::shared_ptr<Constraint>>& GetConstraints() const {
        return this->constraints_;
    }

    /// Returns the number of constraints in the model
    size_t NumConstraints() const { return this->constraints_.size(); }

private:
    std::vector<std::shared_ptr<Node>> nodes_;                 //< Nodes in the model
    std::vector<std::shared_ptr<BeamElement>> beam_elements_;  //< Beam elements in the model
    std::vector<std::shared_ptr<Constraint>> constraints_;     //< Constraints in the model
};

}  // namespace openturbine