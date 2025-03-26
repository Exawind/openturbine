#pragma once

#include "elements/beams/interpolation.hpp"
#include "interfaces/components/blade_input.hpp"
#include "interfaces/node_data.hpp"
#include "math/least_squares_fit.hpp"
#include "math/matrix_operations.hpp"
#include "math/quaternion_operations.hpp"
#include "model/model.hpp"
#include "utilities/scripts/windio_mapped_structs.hpp"

namespace openturbine::interfaces::components {

struct Blade {
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

    Blade(const BladeInput& input, Model& model) : root_node(kInvalidID) {
        // Number of nodes in blade
        const auto n_nodes = input.element_order + 1;

        // Generate node locations within element [-1,1]
        this->node_xi = GenerateGLLPoints(input.element_order);

        // Fit node coordinates to key points
        std::vector<double> kp_xi(MapGeometricLocations(input.ref_axis.coordinate_grid));
        const auto [phi_kn, phi_prime_kn] = ShapeFunctionMatrices(kp_xi, this->node_xi);
        const auto node_coordinates =
            PerformLeastSquaresFitting(n_nodes, phi_kn, input.ref_axis.coordinates);

        // Calculate the derivative shape function matrix for the nodes
        const auto [phi_nn, phi_prime_nn] = ShapeFunctionMatrices(this->node_xi, this->node_xi);

        // Calculate tangent vectors for each node
        std::vector<Array_3> node_tangents(n_nodes, {0., 0., 0.});
        for (auto i = 0U; i < n_nodes; ++i) {
            for (auto j = 0U; j < 3; ++j) {
                for (auto k = 0U; k < n_nodes; ++k) {
                    node_tangents[i][j] += phi_prime_nn[k][i] * node_coordinates[k][j];
                }
            }
        }
        for (auto& node_tangent : node_tangents) {
            double norm = Norm(node_tangent);
            for (auto& v : node_tangent) {
                v /= norm;
            }
        }

        // Add nodes to model
        std::vector<size_t> node_ids;
        for (auto i = 0U; i < this->node_xi.size(); ++i) {
            const auto& pos = node_coordinates[i];
            const auto q_rot = TangentTwistToQuaternion(node_tangents[i], 0.);
            const auto node_id =
                model.AddNode()
                    .SetElemLocation((this->node_xi[i] + 1.) / 2.)
                    .SetPosition(pos[0], pos[1], pos[2], q_rot[0], q_rot[1], q_rot[2], q_rot[3])
                    .Build();
            node_ids.emplace_back(node_id);
            this->nodes.emplace_back(node_id);
        }

        // Extraction section stiffness and mass matrices from blade definition
        std::vector<BeamSection> sections;

        // Add first section after rotating matrices to account for twist
        std::vector<double> section_grid{input.sections[0].location};
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
                section_grid.emplace_back(grid_value);

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
            section_grid.emplace_back(input.sections[i].location);
        }

        // Calculate trapezoidal quadrature based on section locations
        const auto trapezoidal_quadrature = CreateTrapezoidalQuadrature(section_grid);

        // Add beam element and get ID
        this->beam_element_id = model.AddBeamElement(node_ids, sections, trapezoidal_quadrature);

        //----------------------------------------------------------------------
        // Position beam and apply root velocity and acceleration
        //----------------------------------------------------------------------

        const Array_3 root_location{
            input.root.position[0], input.root.position[1], input.root.position[2]
        };
        const Array_4 root_orientation{
            input.root.position[3], input.root.position[4], input.root.position[5],
            input.root.position[6]
        };
        const Array_3 root_omega{
            input.root.velocity[3], input.root.velocity[4], input.root.velocity[5]
        };

        model.TranslateBeam(this->beam_element_id, root_location);
        model.RotateBeamAboutPoint(this->beam_element_id, root_orientation, root_location);
        model.SetBeamVelocityAboutPoint(this->beam_element_id, input.root.velocity, root_location);
        model.SetBeamAccelerationAboutPoint(
            this->beam_element_id, input.root.acceleration, root_omega, root_location
        );

        //----------------------------------------------------------------------
        // Add blade root node and connect to first blade node
        //----------------------------------------------------------------------

        // Add root node
        this->root_node.id = model.AddNode().SetPosition(input.root.position).Build();

        // Constraint first blade node to root node
        this->root_blade_constraint_id =
            model.AddRigidJointConstraint({this->root_node.id, this->nodes[0].id});

        //----------------------------------------------------------------------
        // Add prescribed displacement constraint to root node if requested
        //----------------------------------------------------------------------

        if (input.root.prescribe_root_motion) {
            this->prescribed_root_constraint_id = model.AddPrescribedBC(this->root_node.id);
        }
    }

    /// @brief Return a vector of weights for distributing a point load to the nodes
    /// based on the position [0,1] of the point along the blade
    std::vector<double> GetNodeWeights(double s) {
        std::vector<double> weights(this->node_xi.size());
        auto xi = 2. * s - 1.;
        LagrangePolynomialDerivWeights(xi, this->node_xi, weights);
        return weights;
    }

    /// @brief Add a point load (Fx, Fy, Fz, Mx, My, Mz) to the blade at location 's' [0, 1]
    /// along the material axis
    void AddPointLoad(double s, Array_6 loads) {
        const auto weights = this->GetNodeWeights(s);
        for (size_t i = 0U; i < this->nodes.size(); ++i) {
            for (size_t j = 0U; j < 6; ++j) {
                this->nodes[i].loads[j] += weights[i] * loads[j];
            }
        }
    }

    /// @brief Set blade point loads to zero
    void ClearLoads() {
        for (auto& node : this->nodes) {
            node.ClearLoads();
        }
    }
};

}  // namespace openturbine::interfaces::components
