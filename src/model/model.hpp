#pragma once

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
#include "elements/beams/create_beams.hpp"
#include "elements/elements.hpp"
#include "elements/masses/create_masses.hpp"
#include "elements/masses/masses_input.hpp"
#include "elements/springs/create_springs.hpp"
#include "elements/springs/springs_input.hpp"
#include "mesh_connectivity.hpp"
#include "node.hpp"
#include "solver/solver.hpp"
#include "state/state.hpp"

namespace openturbine {

/**
 * @brief Compute freedom tables for state, elements, and constraints, then construct and return
 * solver.
 *
 * @tparam DeviceType A Kokkos device or execution/memory space
 *
 * @param state A fully initialized State object
 * @param elements  A fully initialized Elements object
 * @param constraints A fully initialized Constraints object
 * @return A solver based on the system connectivity described by the inputs
 */
template <
    typename DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>>
[[nodiscard]] inline Solver<DeviceType> CreateSolver(
    State<DeviceType>& state, Elements<DeviceType>& elements, Constraints<DeviceType>& constraints
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
    /// Represents an invalid node in constraints that only uses the target node
    static constexpr size_t InvalidNodeID{0U};

    /// Default constructor
    Model() = default;

    /**
     * @brief Constructor with gravity specified
     *
     * @param gravity The gravity vector that will be applied during the simulation
     */
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
     * @brief Adds a node to the model
     *
     * @return NodeBuilder object wrapping the newly added node
     */
    NodeBuilder AddNode() {
        const auto id = this->nodes_.size();
        this->nodes_.emplace_back(id);
        return NodeBuilder(this->nodes_.back());
    }

    /**
     * @brief Returns a node by ID
     *
     * @param id Index number of node
     */
    [[nodiscard]] const Node& GetNode(size_t id) const { return this->nodes_[id]; }

    /**
     * @brief Returns a node by ID
     *
     * @param id Index number of node
     */
    [[nodiscard]] Node& GetNode(size_t id) { return this->nodes_[id]; }

    /**
     * @brief Gets the number of nodes present in the model
     *
     * @return The number of nodes
     */
    [[nodiscard]] size_t NumNodes() const { return this->nodes_.size(); }

    /**
     * @brief Returns constant reference to nodes vector
     *
     * @return A reference to the vector containing all of the nodes in the model
     */
    [[nodiscard]] const std::vector<Node>& GetNodes() const { return this->nodes_; }

    //--------------------------------------------------------------------------
    // Beam Elements
    //--------------------------------------------------------------------------

    /**
     * @brief Adds a beam element to the model
     *
     * @param node_ids A list of the node IDs to be contained in the beam
     * @param sections The physical properties defined at each quadrature point
     * @param quadrature The quadrature point locations and weights
     *
     * @return the index of the newly added beam
     */
    size_t AddBeamElement(
        const std::vector<size_t>& node_ids, const std::vector<BeamSection>& sections,
        const std::vector<std::array<double, 2>>& quadrature
    ) {
        const auto elem_id = this->beam_elements_.size();
        this->beam_elements_.emplace_back(elem_id, node_ids, sections, quadrature);
        this->mesh_connectivity_.AddBeamElementConnectivity(elem_id, node_ids);
        return elem_id;
    }

    /**
     * @brief Returns a beam element by ID
     *
     * @param id The index of the beam element
     *
     * @return The beam element itself
     */
    [[nodiscard]] const BeamElement& GetBeamElement(size_t id) const {
        return this->beam_elements_[id];
    }

    /**
     * @brief Returns a beam element by ID
     *
     * @param id The index of the beam element
     *
     * @return The beam element itself
     */
    [[nodiscard]] BeamElement& GetBeamElement(size_t id) { return this->beam_elements_[id]; }

    /**
     * @brief Returns a reference to the beam elements present in the model
     *
     * @return a reference to the vector containing the beam elements
     */
    [[nodiscard]] const std::vector<BeamElement>& GetBeamElements() const {
        return this->beam_elements_;
    }

    /**
     * @brief Returns the number of beam elements present in the model
     *
     * @return the number of beam elements
     */
    [[nodiscard]] size_t NumBeamElements() const { return this->beam_elements_.size(); }

    /**
     * @brief Createsa Beams input file based on the beam elements in the model
     *
     * @return An initialized BeamsInput struct
     */
    [[nodiscard]] BeamsInput CreateBeamsInput() const {
        return {this->beam_elements_, this->gravity_};
    }

    /**
     * @brief Createsa Beams structure based on the beam elements in the model
     *
     * @tparam A Kokkos Device where the Beams struct will exist
     *
     * @return An initialized Beams struct
     */
    template <typename DeviceType>
    [[nodiscard]] Beams<DeviceType> CreateBeams() const {
        return openturbine::CreateBeams<DeviceType>(this->CreateBeamsInput(), this->nodes_);
    }

    /**
     * @brief Translate all beam nodes by given displacement
     *
     * @param beam_elem_id The index of the beam to be translated
     * @param displacement The displacement vector
     */
    void TranslateBeam(size_t beam_elem_id, const std::array<double, 3>& displacement) {
        const auto& beam_elem = this->beam_elements_[beam_elem_id];
        for (const auto& node_id : beam_elem.node_ids) {
            this->GetNode(node_id).Translate(displacement);
        }
    }

    /**
     * @brief Rotate all beam nodes by given quaternion about a given point
     *
     * @param beam_elem_id The index of the beam to be rotated
     * @param displacement_quaternion The displacement quaternion
     * @param point The point around which the beam will be rotated
     */
    void RotateBeamAboutPoint(
        size_t beam_elem_id, const std::array<double, 4>& displacement_quaternion,
        const std::array<double, 3>& point
    ) {
        const auto& beam_elem = this->beam_elements_[beam_elem_id];
        for (const auto& node_id : beam_elem.node_ids) {
            this->GetNode(node_id).RotateAboutPoint(displacement_quaternion, point);
        }
    }

    /**
     * @brief Set the translational and rotational velocity of the beam about a given point
     *
     * @param beam_elem_id The index of the beam
     * @param velocity The velocity of the beam
     * @param point The point about which the rotational velocity is based
     */
    void SetBeamVelocityAboutPoint(
        size_t beam_elem_id, const std::array<double, 6>& velocity,
        const std::array<double, 3>& point
    ) {
        const auto& beam_elem = this->beam_elements_[beam_elem_id];
        for (const auto& node_id : beam_elem.node_ids) {
            this->GetNode(node_id).SetVelocityAboutPoint(velocity, point);
        }
    }

    /**
     * @brief Set the acceleration of the beam about a given point
     *
     * @param beam_elem_id The index of the beam
     * @param acceleration The acceleration of the beam
     * @param omega The rotational acceleration of the beam
     * @param point The point about which the rotational velocity is based
     */
    void SetBeamAccelerationAboutPoint(
        size_t beam_elem_id, const std::array<double, 6>& acceleration,
        const std::array<double, 3>& omega, const std::array<double, 3>& point
    ) {
        const auto& beam_elem = this->beam_elements_[beam_elem_id];
        for (const auto& node_id : beam_elem.node_ids) {
            this->GetNode(node_id).SetAccelerationAboutPoint(acceleration, omega, point);
        }
    }

    //--------------------------------------------------------------------------
    // Mass Elements
    //--------------------------------------------------------------------------

    /**
     * @brief Adds a mass element to the model
     *
     * @param node_id ID of the node where the mass element will be placed
     * @param mass The inertia matrix of the element
     *
     * @return The index of the newly added element
     */
    size_t AddMassElement(const size_t node_id, const std::array<std::array<double, 6>, 6>& mass) {
        const auto elem_id = this->mass_elements_.size();
        this->mass_elements_.emplace_back(elem_id, node_id, mass);
        this->mesh_connectivity_.AddMassElementConnectivity(elem_id, node_id);
        return elem_id;
    }

    /**
     * @brief Returns a mass element by ID
     *
     * @param id ID of desired mass element
     *
     * @return The Mass element
     */
    [[nodiscard]] const MassElement& GetMassElement(size_t id) const {
        return this->mass_elements_[id];
    }

    /**
     * @brief Returns a mass element by ID
     *
     * @param id ID of desired mass element
     *
     * @return The Mass element
     */
    [[nodiscard]] MassElement& GetMassElement(size_t id) { return this->mass_elements_[id]; }

    /**
     * @brief Returns a reference to the mass elements present in the model
     *
     * @return A reference to the vector containing the mass elements
     */
    [[nodiscard]] const std::vector<MassElement>& GetMassElements() const {
        return this->mass_elements_;
    }

    /**
     * @brief Returns the number of mass elements present in the model
     *
     * @return The number of mass elements
     */
    [[nodiscard]] size_t NumMassElements() const { return this->mass_elements_.size(); }

    /**
     * @brief Create a a masses struct based on the mass elements present in the model
     *
     * @tparam DeviceType A Kokkos device where the Masses object will reside
     *
     * @return an initialized Masses object
     */
    template <typename DeviceType>
    [[nodiscard]] Masses<DeviceType> CreateMasses() const {
        return openturbine::CreateMasses<DeviceType>(
            MassesInput(this->mass_elements_, this->gravity_), this->nodes_
        );
    }

    //--------------------------------------------------------------------------
    // Spring Elements
    //--------------------------------------------------------------------------

    /**
     * @brief Adds a spring element to the model
     *
     * @param node1_id ID of the node at one end of the spring
     * @param node2_id ID of the node at the other end of the spring
     * @param stiffness Stiffness of the spring
     * @param undeformed_length Length of the spring at which the spring force is zero
     *
     * @return the index of the newly added spring
     */
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

    /**
     * @brief Returns a spring element by ID
     *
     * @param id The ID of the spring
     *
     * @return The requested spring element
     */
    [[nodiscard]] const SpringElement& GetSpringElement(size_t id) const {
        return this->spring_elements_[id];
    }

    /**
     * @brief Returns a spring element by ID
     *
     * @param id The ID of the spring
     *
     * @return The requested spring element
     */
    [[nodiscard]] SpringElement& GetSpringElement(size_t id) { return this->spring_elements_[id]; }

    /**
     * @brief Returns the number of spring elements present in the model
     *
     * @return the number of springs in the model
     */
    [[nodiscard]] size_t NumSpringElements() const { return this->spring_elements_.size(); }

    /**
     * @brief Creates a Springs struct based on the spring elements in the model
     *
     * @tparam DeviceType a Kokkos device where the Springs object will reside
     *
     * @return An initialized Springs object
     */
    template <typename DeviceType>
    [[nodiscard]] Springs<DeviceType> CreateSprings() const {
        return openturbine::CreateSprings<DeviceType>(
            SpringsInput(this->spring_elements_), this->nodes_
        );
    }

    //--------------------------------------------------------------------------
    // Elements
    //--------------------------------------------------------------------------

    /**
     * @brief Creates an Elements struct with Beams, Masses, and Springs
     *
     * @tparam DeviceType a Kokkos device where the Elements object will reside
     *
     * @return an initialized Elements object
     */
    template <typename DeviceType>
    [[nodiscard]] Elements<DeviceType> CreateElements() const {
        return {
            this->CreateBeams<DeviceType>(),
            this->CreateMasses<DeviceType>(),
            this->CreateSprings<DeviceType>(),
        };
    }

    //--------------------------------------------------------------------------
    // State
    //--------------------------------------------------------------------------

    /**
     * @brief Creates an State struct based on the nodes in this model
     *
     * @tparam DeviceType a Kokkos device where the State object will reside
     *
     * @return an initialized State object
     */
    template <typename DeviceType>
    [[nodiscard]] State<DeviceType> CreateState() const {
        auto state = State<DeviceType>(this->nodes_.size());
        CopyNodesToState<DeviceType>(state, this->nodes_);
        return state;
    }

    //--------------------------------------------------------------------------
    // Constraints
    //--------------------------------------------------------------------------

    /// Adds a fixed boundary condition constraint to the model and returns the ID
    size_t AddFixedBC(const size_t node_id) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::FixedBC, std::array{InvalidNodeID, node_id}
        );
        this->mesh_connectivity_.AddConstraintConnectivity(id, std::vector<size_t>{node_id});
        return id;
    }

    /// Adds a prescribed boundary condition constraint to the model and returns the ID
    size_t AddPrescribedBC(
        const size_t node_id,
        const std::array<double, 7>& initial_displacement = {0., 0., 0., 1., 0., 0., 0.}
    ) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::PrescribedBC, std::array{InvalidNodeID, node_id},
            std::array{0., 0., 0.}, initial_displacement
        );
        this->mesh_connectivity_.AddConstraintConnectivity(id, std::vector<size_t>{node_id});
        return id;
    }

    /// Adds a rigid constraint to the model and returns the ID
    size_t AddRigidJointConstraint(const std::array<size_t, 2>& node_ids) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(id, ConstraintType::RigidJoint, node_ids);
        this->mesh_connectivity_.AddConstraintConnectivity(
            id, std::vector<size_t>{node_ids[0], node_ids[1]}
        );
        return id;
    }

    /// Adds a revolute/hinge constraint to the model and returns the ID
    size_t AddRevoluteJointConstraint(
        const std::array<size_t, 2>& node_ids, const std::array<double, 3>& axis, double* torque
    ) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::RevoluteJoint, node_ids, axis,
            std::array{0., 0., 0., 1., 0., 0., 0.}, torque
        );
        this->mesh_connectivity_.AddConstraintConnectivity(
            id, std::vector<size_t>{node_ids[0], node_ids[1]}
        );
        return id;
    }

    /// Adds a rotation control constraint to the model and returns the ID
    size_t AddRotationControl(
        const std::array<size_t, 2>& node_ids, const std::array<double, 3>& axis, double* control
    ) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::RotationControl, node_ids, axis,
            std::array{0., 0., 0., 1., 0., 0., 0.}, control
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
            id, ConstraintType::FixedBC3DOFs, std::array{InvalidNodeID, node_id}
        );
        this->mesh_connectivity_.AddConstraintConnectivity(id, std::vector<size_t>{node_id});
        return id;
    }

    /// Adds a prescribed boundary condition constraint (6DOFs to 3DOFs) to the model and returns
    /// the ID
    size_t AddPrescribedBC3DOFs(const size_t node_id) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(
            id, ConstraintType::PrescribedBC3DOFs, std::array{InvalidNodeID, node_id}
        );
        this->mesh_connectivity_.AddConstraintConnectivity(id, std::vector<size_t>{node_id});
        return id;
    }

    /// Adds a rigid joint constraint (6DOFs to 3DOFs) to the model and returns the ID
    size_t AddRigidJoint6DOFsTo3DOFs(const std::array<size_t, 2>& node_ids) {
        const auto id = this->constraints_.size();
        this->constraints_.emplace_back(id, ConstraintType::RigidJoint6DOFsTo3DOFs, node_ids);
        this->mesh_connectivity_.AddConstraintConnectivity(
            id, std::vector<size_t>{node_ids[0], node_ids[1]}
        );
        return id;
    }

    /// Returns the number of constraints present in the model
    [[nodiscard]] size_t NumConstraints() const { return this->constraints_.size(); }

    /// Returns a Constraints object initialized from the model constraints
    template <typename DeviceType>
    [[nodiscard]] Constraints<DeviceType> CreateConstraints() const {
        return Constraints<DeviceType>(this->constraints_, this->nodes_);
    }

    /// Returns a State, Elements, and Constraints object initialized from the model
    template <
        typename DeviceType = Kokkos::Device<
            Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>>
    [[nodiscard]] std::tuple<State<DeviceType>, Elements<DeviceType>, Constraints<DeviceType>>
    CreateSystem() const {
        return {
            this->CreateState<DeviceType>(), this->CreateElements<DeviceType>(),
            this->CreateConstraints<DeviceType>()
        };
    }

    /// Returns a State, Elements, Constraints, and Solver object initialized from the model
    template <
        typename DeviceType = Kokkos::Device<
            Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>>
    [[nodiscard]] std::tuple<
        State<DeviceType>, Elements<DeviceType>, Constraints<DeviceType>, Solver<DeviceType>>
    CreateSystemWithSolver() const {
        auto [state, elements, constraints] = this->CreateSystem<DeviceType>();
        auto solver = CreateSolver<DeviceType>(state, elements, constraints);
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
    std::array<double, 3> gravity_ = {0., 0., 0.};  //< Gravity components
    std::vector<Node> nodes_;                       //< Nodes in the model
    std::vector<BeamElement> beam_elements_;        //< Beam elements in the model
    std::vector<MassElement> mass_elements_;        //< Mass elements in the model
    std::vector<SpringElement> spring_elements_;    //< Spring elements in the model
    std::vector<Constraint> constraints_;           //< Constraints in the model
    MeshConnectivity mesh_connectivity_;  //< Mesh connectivity tracking element-node relationships
};

}  // namespace openturbine
