#pragma once

#include "elements/beams/interpolation.hpp"
#include "interfaces/components/beam_input.hpp"
#include "interfaces/node_data.hpp"
#include "math/least_squares_fit.hpp"
#include "math/matrix_operations.hpp"
#include "math/project_points_to_target_polynomial.hpp"
#include "math/quaternion_operations.hpp"
#include "model/model.hpp"

namespace openturbine::interfaces::components {

/**
 * @brief Represents a turbine blade with nodes, elements, and constraints
 *
 * This class is responsible for creating and managing a blade based on input
 * specifications. It handles the creation of nodes, beam elements, and constraints
 * within the provided model.
 */
class Beam {
public:
    /// @brief Maximum number of points allowed in blade geometry definition
    static constexpr size_t kMaxGeometryPoints{10};

    /// @brief Beam element ID
    size_t beam_element_id{kInvalidID};

    /// @brief Blade node data
    std::vector<NodeData> nodes;

    /// @brief Blade root node
    NodeData root_node;

    /// @brief ID of constraint connecting root node to first blade node
    size_t root_blade_constraint_id{kInvalidID};

    /// @brief Constraint ID of prescribed root displacement
    size_t prescribed_root_constraint_id{kInvalidID};

    /// @brief Location of nodes in blade element [-1, 1]
    std::vector<double> node_xi;

    /**
     * @brief Construct a new Blade using the provided input and model
     * @param input Configuration for the blade
     * @param model Model to which the blade elements and nodes will be added
     * @throws std::invalid_argument If the input configuration is invalid
     */
    Beam(const BeamInput& input, Model& model) : root_node(kInvalidID) {
        ValidateInput(input);
        SetupNodeLocations(input);
        CreateNodeGeometry(input);
        CreateBeamElement(input, model);
        PositionBladeInSpace(input, model);
        SetupRootNode(input, model);
    }

    /**
     * @brief Returns a vector of weights for distributing a point load to the nodes
     * @param s Position [0,1] along the blade
     * @return Vector of weights for each node
     */
    [[nodiscard]] std::vector<double> GetNodeWeights(double s) const {
        std::vector<double> weights(this->node_xi.size());
        auto xi = 2. * s - 1.;
        LagrangePolynomialDerivWeights(xi, this->node_xi, weights);
        return weights;
    }

    /**
     * @brief Adds a point load (Fx, Fy, Fz, Mx, My, Mz) to the blade at location 's' [0, 1]
     * along the material axis
     * @param s Position [0,1] along the blade material axis
     * @param loads Forces and moments (Fx, Fy, Fz, Mx, My, Mz)
     * @throws std::invalid_argument If position is outside valid range
     */
    void AddPointLoad(double s, std::array<double, 6> loads) {
        if (s < 0. || s > 1.) {
            throw std::invalid_argument("Invalid position: " + std::to_string(s));
        }

        const auto weights = this->GetNodeWeights(s);
        for (size_t i = 0U; i < this->nodes.size(); ++i) {
            for (size_t j = 0U; j < 6; ++j) {
                this->nodes[i].loads[j] += weights[i] * loads[j];
            }
        }
    }

    /// @brief Sets blade point loads to zero
    void ClearLoads() {
        for (auto& node : this->nodes) {
            node.ClearLoads();
        }
    }

private:
    std::vector<std::array<double, 3>> node_coordinates;  ///< Node coordinates
    std::vector<std::array<double, 3>> node_tangents;     ///< Node tangents

    /**
     * @brief Validate the input configuration
     * @param input Blade input configuration
     * @throws std::invalid_argument If configuration is invalid
     */
    static void ValidateInput(const BeamInput& input) {
        if (input.ref_axis.coordinate_grid.size() < 2 || input.ref_axis.coordinates.size() < 2) {
            throw std::invalid_argument("At least two reference axis points are required");
        }
        if (input.ref_axis.coordinate_grid.size() != input.ref_axis.coordinates.size()) {
            throw std::invalid_argument("Mismatch between coordinate_grid and coordinates sizes");
        }
        if (input.sections.empty()) {
            throw std::invalid_argument("At least one section is required");
        }
        if (input.element_order < 1) {
            throw std::invalid_argument(
                "Element order must be at least 1 i.e. linear element for discretization"
            );
        }
    }

    /**
     * @brief Setup node locations based on input configuration
     * @param input Blade input configuration
     */
    void SetupNodeLocations(const BeamInput& input) {
        // Generate node locations within element [-1,1]
        this->node_xi = GenerateGLLPoints(input.element_order);
    }

    /**
     * @brief Create node geometry from reference axis points
     * @param input Blade input configuration
     */
    void CreateNodeGeometry(const BeamInput& input) {
        const auto n_nodes = input.element_order + 1;
        const auto n_geometry_pts =
            std::min({input.ref_axis.coordinate_grid.size(), n_nodes, kMaxGeometryPoints});

        if (n_geometry_pts < n_nodes) {
            // We need to project from n_geometry_pts -> element_order
            const std::vector<double> kp_xi(MapGeometricLocations(input.ref_axis.coordinate_grid));
            const auto [phi_kn_geometry, phi_prime_kn_geometry] =
                ShapeFunctionMatrices(kp_xi, GenerateGLLPoints(n_geometry_pts - 1));
            const auto geometry_points = PerformLeastSquaresFitting(
                n_geometry_pts, phi_kn_geometry, input.ref_axis.coordinates
            );
            const auto node_coords =
                ProjectPointsToTargetPolynomial(n_geometry_pts, n_nodes, geometry_points);

            this->node_coordinates.clear();
            this->node_coordinates.reserve(node_coords.size());
            std::copy(
                node_coords.begin(), node_coords.end(), std::back_inserter(this->node_coordinates)
            );
        } else {
            // Fit node coordinates to key points
            const std::vector<double> kp_xi(MapGeometricLocations(input.ref_axis.coordinate_grid));
            const auto [phi_kn, phi_prime_kn] = ShapeFunctionMatrices(kp_xi, this->node_xi);
            this->node_coordinates =
                PerformLeastSquaresFitting(n_nodes, phi_kn, input.ref_axis.coordinates);
        }

        // Calculate tangent vectors at each node
        this->CalcNodeTangents();
    }

    /**
     * @brief Create beam element in the model
     * @param input Blade input configuration
     * @param model Model to which the beam element will be added
     */
    void CreateBeamElement(const BeamInput& input, Model& model) {
        // Add nodes to model
        std::vector<size_t> node_ids;
        for (auto i = 0U; i < this->node_xi.size(); ++i) {
            const auto& pos = this->node_coordinates[i];
            const auto q_rot = TangentTwistToQuaternion(this->node_tangents[i], 0.);
            const auto node_id =
                model.AddNode()
                    .SetElemLocation((this->node_xi[i] + 1.) / 2.)
                    .SetPosition(pos[0], pos[1], pos[2], q_rot[0], q_rot[1], q_rot[2], q_rot[3])
                    .Build();
            node_ids.emplace_back(node_id);
            this->nodes.emplace_back(node_id);
        }

        // Build beam sections
        const auto sections = BuildBeamSections(input);
        std::vector<double> section_grid(sections.size());
        std::transform(
            sections.begin(), sections.end(), section_grid.begin(),
            [](const auto& section) {
                return section.position;
            }
        );

        // Calculate trapezoidal quadrature based on section locations
        const auto trapezoidal_quadrature = CreateTrapezoidalQuadrature(section_grid);

        // Add beam element and get ID
        this->beam_element_id = model.AddBeamElement(node_ids, sections, trapezoidal_quadrature);
    }

    /**
     * @brief Position the blade in space according to root properties
     * @param input Blade input configuration
     * @param model Model containing the blade
     */
    void PositionBladeInSpace(const BeamInput& input, Model& model) const {
        // Extract root location, orientation, and velocity
        const std::array<double, 3> root_location{
            input.root.position[0], input.root.position[1], input.root.position[2]
        };
        const Array_4 root_orientation{
            input.root.position[3], input.root.position[4], input.root.position[5],
            input.root.position[6]
        };
        const std::array<double, 3> root_omega{
            input.root.velocity[3], input.root.velocity[4], input.root.velocity[5]
        };

        // Translate beam element to root location
        model.TranslateBeam(this->beam_element_id, root_location);

        // Rotate beam element about root location
        model.RotateBeamAboutPoint(this->beam_element_id, root_orientation, root_location);

        // Set beam velocity about root location
        model.SetBeamVelocityAboutPoint(this->beam_element_id, input.root.velocity, root_location);

        // Set beam acceleration about root location
        model.SetBeamAccelerationAboutPoint(
            this->beam_element_id, input.root.acceleration, root_omega, root_location
        );
    }

    /**
     * @brief Setup the root node and constraints
     * @param input Blade input configuration
     * @param model Model to which constraints will be added
     */
    void SetupRootNode(const BeamInput& input, Model& model) {
        // Add root node
        this->root_node.id = model.AddNode().SetPosition(input.root.position).Build();

        // Constraint first blade node to root node
        this->root_blade_constraint_id =
            model.AddRigidJointConstraint({this->root_node.id, this->nodes[0].id});

        // Add prescribed displacement constraint to root node if requested
        if (input.root.prescribe_root_motion) {
            this->prescribed_root_constraint_id = model.AddPrescribedBC(this->root_node.id);
        }
    }

    /**
     * @brief Calculate tangent vectors at each node
     */
    void CalcNodeTangents() {
        const auto n_nodes{this->node_coordinates.size()};

        // Calculate the derivative shape function matrix for the nodes
        const auto [phi, phi_prime] = ShapeFunctionMatrices(this->node_xi, this->node_xi);

        // Calculate tangent vectors for each node
        this->node_tangents.resize(n_nodes, {0., 0., 0.});
        for (auto i = 0U; i < n_nodes; ++i) {
            for (auto j = 0U; j < 3; ++j) {
                for (auto k = 0U; k < n_nodes; ++k) {
                    this->node_tangents[i][j] += phi_prime[k][i] * this->node_coordinates[k][j];
                }
            }
        }

        // Normalize tangent vectors
        std::transform(
            this->node_tangents.begin(), this->node_tangents.end(), this->node_tangents.begin(),
            [](std::array<double, 3>& tangent) {
                const auto norm = Norm(tangent);
                std::transform(tangent.begin(), tangent.end(), tangent.begin(), [norm](double v) {
                    return v / norm;
                });
                return tangent;
            }
        );
    }

    /**
     * @brief Create beam sections from input configuration
     * @param input Blade input configuration
     * @return Vector of beam sections
     */
    static std::vector<BeamSection> BuildBeamSections(const BeamInput& input) {
        // Extraction section stiffness and mass matrices from blade definition
        std::vector<BeamSection> sections;

        // Add first section after rotating matrices to account for twist
        auto twist = LinearInterp(
            input.sections[0].location, input.ref_axis.twist_grid, input.ref_axis.twist
        );
        auto q_twist = RotationVectorToQuaternion({twist * M_PI / 180., 0., 0.});
        sections.emplace_back(
            input.sections[0].location, RotateMatrix6(input.sections[0].mass_matrix, q_twist),
            RotateMatrix6(input.sections[0].stiffness_matrix, q_twist)
        );

        // Loop through remaining section locations
        for (auto i = 1U; i < input.sections.size(); ++i) {
            // Add refinement sections if requested
            for (auto j = 0U; j < input.section_refinement; ++j) {
                // Calculate interpolation ratio between bounding sections
                const auto alpha =
                    static_cast<double>(j + 1) / static_cast<double>(input.section_refinement + 1);

                // Interpolate grid location
                const auto grid_value = (1. - alpha) * input.sections[i - 1].location +
                                        alpha * input.sections[i].location;

                // Interpolate mass and stiffness matrices from bounding sections
                Array_6x6 mass_matrix;
                Array_6x6 stiffness_matrix;
                for (auto mi = 0U; mi < 6; ++mi) {
                    for (auto ni = 0U; ni < 6; ++ni) {
                        mass_matrix[mi][ni] =
                            (1. - alpha) * input.sections[i - 1].mass_matrix[mi][ni] +
                            alpha * input.sections[i].mass_matrix[mi][ni];
                        stiffness_matrix[mi][ni] =
                            (1. - alpha) * input.sections[i - 1].stiffness_matrix[mi][ni] +
                            alpha * input.sections[i].stiffness_matrix[mi][ni];
                    }
                }

                // Calculate twist at current section location via linear interpolation
                twist = LinearInterp(
                    input.sections[i].location, input.ref_axis.twist_grid, input.ref_axis.twist
                );

                // Add refinement section
                q_twist = RotationVectorToQuaternion({twist * M_PI / 180., 0., 0.});
                sections.emplace_back(
                    grid_value, RotateMatrix6(mass_matrix, q_twist),
                    RotateMatrix6(stiffness_matrix, q_twist)
                );
            }

            // Add ending section
            twist = LinearInterp(
                input.sections[i].location, input.ref_axis.twist_grid, input.ref_axis.twist
            );
            q_twist = RotationVectorToQuaternion({twist * M_PI / 180., 0., 0.});
            sections.emplace_back(
                input.sections[i].location, RotateMatrix6(input.sections[i].mass_matrix, q_twist),
                RotateMatrix6(input.sections[i].stiffness_matrix, q_twist)
            );
        }

        return sections;
    }
};

}  // namespace openturbine::interfaces::components
