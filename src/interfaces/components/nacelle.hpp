#pragma once

#include "interfaces/components/nacelle_input.hpp"
#include "interfaces/node_data.hpp"
#include "model/model.hpp"

namespace openturbine::interfaces::components {

/**
 * @brief Represents a turbine blade with nodes, elements, and constraints
 *
 * This class is responsible for creating and managing a blade based on input
 * specifications. It handles the creation of nodes, beam elements, and constraints
 * within the provided model.
 */
class Nacelle {
    NodeData cm_node;           //< Center-of-mass node
    NodeData yaw_bearing_node;  //< Yaw bearing node
    NodeData shaft_base_node;   //< Shaft base node
    NodeData generator_node;    //< Generator node

public:
    /**
     * @brief Construct a new Blade using the provided input and model
     * @param input Configuration for the blade
     * @param model Model to which the blade elements and nodes will be added
     * @throws std::invalid_argument If the input configuration is invalid
     */
    Nacelle(const NacelleInput& input, Model& model)
        : cm_node(kInvalidID),
          yaw_bearing_node(kInvalidID),
          shaft_base_node(kInvalidID),
          generator_node(kInvalidID) {
        ValidateInput(input);
        SetupNodeLocations(input);
        CreateNodeGeometry(input);
        CreateNacelleElement(input, model);
        PositionBladeInSpace(input, model);
        SetupRootNode(input, model);
    }

private:
    /**
     * @brief Validate the input configuration
     * @param input Blade input configuration
     * @throws std::invalid_argument If configuration is invalid
     */
    static void ValidateInput(const NacelleInput& input) {
        // if (input.ref_axis.coordinate_grid.empty() || input.ref_axis.coordinates.empty()) {
        //     throw std::invalid_argument("At least one reference axis point is required");
        // }
    }

    /**
     * @brief Create node geometry from reference axis points
     * @param input Blade input configuration
     */
    void CreateNodes(const NacelleInput& input, Model& model) {}

    /**
     * @brief Create beam element in the model
     * @param input Blade input configuration
     * @param model Model to which the beam element will be added
     */
    void CreateNacelleElement(const NacelleInput& input, Model& model) {

    }

    /**
     * @brief Position the blade in space according to root properties
     * @param input Blade input configuration
     * @param model Model containing the blade
     */
    void PositionBladeInSpace(const NacelleInput& input, Model& model) const {
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
        model.TranslateNacelle(this->beam_element_id, root_location);

        // Rotate beam element about root location
        model.RotateNacelleAboutPoint(this->beam_element_id, root_orientation, root_location);

        // Set beam velocity about root location
        model.SetNacelleVelocityAboutPoint(
            this->beam_element_id, input.root.velocity, root_location
        );

        // Set beam acceleration about root location
        model.SetNacelleAccelerationAboutPoint(
            this->beam_element_id, input.root.acceleration, root_omega, root_location
        );
    }

    /**
     * @brief Setup the root node and constraints
     * @param input Blade input configuration
     * @param model Model to which constraints will be added
     */
    void SetupRootNode(const NacelleInput& input, Model& model) {
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
    static std::vector<NacelleSection> BuildNacelleSections(const NacelleInput& input) {
        // Extraction section stiffness and mass matrices from blade definition
        std::vector<NacelleSection> sections;

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
