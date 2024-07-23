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
struct Model {
    std::vector<Node> nodes;
    std::vector<Constraint> constraints;

    /// Adds a node to the model and returns the node
    Node AddNode(
        const Array_7& position, const Array_7& displacement = Array_7{0., 0., 0., 1., 0., 0., 0.},
        const Array_6& velocity = Array_6{0., 0., 0., 0., 0., 0.},
        const Array_6& acceleration = Array_6{0., 0., 0., 0., 0., 0.}
    ) {
        this->nodes.push_back(Node(nodes.size(), position, displacement, velocity, acceleration));
        return this->nodes.back();
    }

    /// Adds a rigid constraint to the model and returns the constraint.
    Constraint AddRigidConstraint(const Node& node1, const Node& node2) {
        this->constraints.push_back(
            Constraint(ConstraintType::kRigid, this->constraints.size(), node1, node2)
        );
        return this->constraints.back();
    }

    /// Adds a prescribed boundary condition constraint to the model and returns the constraint.
    Constraint AddPrescribedBC(const Node& node, const Array_3& ref_position = {0., 0., 0.}) {
        this->constraints.push_back(Constraint(
            ConstraintType::kPrescribedBC, this->constraints.size(), InvalidNode, node, ref_position
        ));
        return this->constraints.back();
    }

    /// Adds a fixed boundary condition constraint to the model and returns the constraint.
    Constraint AddFixedBC(const Node& node) {
        this->constraints.push_back(
            Constraint(ConstraintType::kFixedBC, this->constraints.size(), InvalidNode, node)
        );
        return this->constraints.back();
    }

    /// Adds a cylindrical constraint to the model and returns the constraint.
    Constraint AddCylindricalConstraint(const Node& node1, const Node& node2) {
        this->constraints.push_back(
            Constraint(ConstraintType::kCylindrical, this->constraints.size(), node1, node2)
        );
        return this->constraints.back();
    }

    /// Adds a rotation control constraint to the model and returns the constraint.
    Constraint AddRotationControl(
        const Node& node1, const Node& node2, const Array_3& axis, float* control
    ) {
        this->constraints.push_back(Constraint(
            ConstraintType::kRotationControl, this->constraints.size(), node1, node2, axis, control
        ));
        return this->constraints.back();
    }

    /// Returns the number of nodes in the model
    int NumNodes() { return this->nodes.size(); }
};

class Model_2 {
public:
    Model_2() = default;

    Model_2(std::vector<Node> nodes, std::vector<BeamElement> beam_elements) {
        for (auto& node : nodes) {
            this->nodes_.push_back(std::make_shared<Node>(node));
        }
        for (auto& element : beam_elements) {
            this->beam_elements_.push_back(std::make_shared<BeamElement>(element));
        }
    }

    Model_2(
        std::vector<std::shared_ptr<Node>> nodes,
        std::vector<std::shared_ptr<BeamElement>> beam_elements
    )
        : nodes_(std::move(nodes)), beam_elements_(std::move(beam_elements)) {}

    // Make the class non-copyable and non-assignable
    Model_2(const Model_2&) = delete;
    Model_2& operator=(const Model_2&) = delete;

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

    /// Returns a reference to the nodes in the model
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

private:
    std::vector<std::shared_ptr<Node>> nodes_;                 //< Nodes in the model
    std::vector<std::shared_ptr<BeamElement>> beam_elements_;  //< Beam elements in the model
};

}  // namespace openturbine