#include "beam.hpp"

#include <stdexcept>

#include "elements/beams/beam_quadrature.hpp"
#include "elements/beams/interpolation.hpp"
#include "interfaces/components/beam_input.hpp"
#include "math/least_squares_fit.hpp"
#include "math/matrix_operations.hpp"
#include "math/project_points_to_target_polynomial.hpp"
#include "math/quaternion_operations.hpp"
#include "model/model.hpp"

namespace openturbine::interfaces::components {

Beam::Beam(const BeamInput& input, Model& model) {
    ValidateInput(input);
    SetupNodeLocations(input);
    CreateNodeGeometry(input);
    CreateBeamElement(input, model);
    PositionBladeInSpace(input, model);
    SetupRootNode(input, model);
}

std::vector<double> Beam::GetNodeWeights(double s) const {
    std::vector<double> weights(this->node_xi.size());
    auto xi = 2. * s - 1.;
    LagrangePolynomialDerivWeights(xi, this->node_xi, weights);
    return weights;
}

void Beam::AddPointLoad(double s, std::array<double, 6> loads) {
    if (s < 0. || s > 1.) {
        throw std::invalid_argument("Invalid position: " + std::to_string(s));
    }

    const auto weights = this->GetNodeWeights(s);
    for (auto node = 0U; node < this->nodes.size(); ++node) {
        for (auto component = 0U; component < 6U; ++component) {
            this->nodes[node].loads[component] += weights[node] * loads[component];
        }
    }
}

void Beam::ClearLoads() {
    for (auto& node : this->nodes) {
        node.ClearLoads();
    }
}

void Beam::GetMotion(const HostState<DeviceType>& host_state) {
    for (auto& node : this->nodes) {
        node.GetMotion(host_state);
    }
}

void Beam::SetLoads(HostState<DeviceType>& host_state) const {
    for (const auto& node : this->nodes) {
        node.SetLoads(host_state);
    }
}

void Beam::ValidateInput(const BeamInput& input) {
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

void Beam::SetupNodeLocations(const BeamInput& input) {
    // Generate node locations within element [-1,1]
    this->node_xi = GenerateGLLPoints(input.element_order);
}

void Beam::CreateNodeGeometry(const BeamInput& input) {
    const auto n_nodes = input.element_order + 1;
    const auto n_geometry_pts =
        std::min({input.ref_axis.coordinate_grid.size(), n_nodes, kMaxGeometryPoints});

    if (n_geometry_pts < n_nodes) {
        // We need to project from n_geometry_pts -> element_order
        const std::vector<double> kp_xi(MapGeometricLocations(input.ref_axis.coordinate_grid));
        const auto gll_points = GenerateGLLPoints(n_geometry_pts - 1);
        const auto phi_kn_geometry = ComputeShapeFunctionValues(kp_xi, gll_points);
        const auto geometry_points =
            PerformLeastSquaresFitting(n_geometry_pts, phi_kn_geometry, input.ref_axis.coordinates);
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
        const auto phi_kn = ComputeShapeFunctionValues(kp_xi, this->node_xi);
        this->node_coordinates =
            PerformLeastSquaresFitting(n_nodes, phi_kn, input.ref_axis.coordinates);
    }

    // Calculate tangent vectors at each node
    this->CalcNodeTangents();
}

void Beam::CreateBeamElement(const BeamInput& input, Model& model) {
    // Add nodes to model
    std::vector<size_t> node_ids;
    for (auto node = 0U; node < this->node_xi.size(); ++node) {
        const auto& pos = this->node_coordinates[node];
        const auto q_rot = TangentTwistToQuaternion(this->node_tangents[node], 0.);
        const auto node_id =
            model.AddNode()
                .SetElemLocation((this->node_xi[node] + 1.) / 2.)
                .SetPosition(pos[0], pos[1], pos[2], q_rot[0], q_rot[1], q_rot[2], q_rot[3])
                .Build();
        node_ids.emplace_back(node_id);
        this->nodes.emplace_back(node_id);
    }

    // Build beam sections
    const auto sections = BuildBeamSections(input);
    std::vector<double> section_grid(sections.size());
    std::transform(sections.begin(), sections.end(), section_grid.begin(), [](const auto& section) {
        return section.position;
    });

    // Calculate trapezoidal quadrature based on section locations
    const auto trapezoidal_quadrature = CreateTrapezoidalQuadrature(section_grid);

    // Add beam element and get ID
    this->beam_element_id = model.AddBeamElement(node_ids, sections, trapezoidal_quadrature);
}

void Beam::PositionBladeInSpace(const BeamInput& input, Model& model) const {
    // Extract root location, orientation, and velocity
    const auto root_location =
        std::array{input.root.position[0], input.root.position[1], input.root.position[2]};
    const auto root_orientation = std::array{
        input.root.position[3], input.root.position[4], input.root.position[5],
        input.root.position[6]
    };
    const auto root_omega =
        std::array{input.root.velocity[3], input.root.velocity[4], input.root.velocity[5]};

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

void Beam::SetupRootNode(const BeamInput& input, Model& model) {
    // Add prescribed displacement constraint to root node if requested
    if (input.root.prescribe_root_motion) {
        this->prescribed_root_constraint_id = model.AddPrescribedBC(this->nodes[0].id);
    }
}

void Beam::CalcNodeTangents() {
    const auto n_nodes{this->node_coordinates.size()};

    // Calculate the derivative shape function matrix for the nodes
    const auto phi = ComputeShapeFunctionValues(this->node_xi, this->node_xi);
    const auto phi_prime = ComputeShapeFunctionDerivatives(this->node_xi, this->node_xi);

    // Calculate tangent vectors for each node
    this->node_tangents.resize(n_nodes, {0., 0., 0.});
    for (auto node_1 = 0U; node_1 < n_nodes; ++node_1) {
        for (auto node_2 = 0U; node_2 < n_nodes; ++node_2) {
            const auto shape_deriv = phi_prime[node_1][node_2];
            for (auto component = 0U; component < 3; ++component) {
                this->node_tangents[node_2][component] +=
                    shape_deriv * this->node_coordinates[node_1][component];
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

std::vector<BeamSection> Beam::BuildBeamSections(const BeamInput& input) {
    // Extraction section stiffness and mass matrices from blade definition
    std::vector<BeamSection> sections;

    // Add first section after rotating matrices to account for twist
    auto twist =
        LinearInterp(input.sections[0].location, input.ref_axis.twist_grid, input.ref_axis.twist);
    auto q_twist = RotationVectorToQuaternion({twist * M_PI / 180., 0., 0.});
    sections.emplace_back(
        input.sections[0].location, RotateMatrix6(input.sections[0].mass_matrix, q_twist),
        RotateMatrix6(input.sections[0].stiffness_matrix, q_twist)
    );

    // Loop through remaining section locations
    for (auto section = 1U; section < input.sections.size(); ++section) {
        const auto section_location = input.sections[section].location;
        const auto section_mass_matrix = input.sections[section].mass_matrix;
        const auto section_stiffness_matrix = input.sections[section].stiffness_matrix;

        // Add refinement sections if requested
        for (auto refinement = 0U; refinement < input.section_refinement; ++refinement) {
            const auto left_location = input.sections[section - 1].location;
            const auto left_mass_matrix = input.sections[section - 1].mass_matrix;
            const auto left_stiffness_matrix = input.sections[section - 1].stiffness_matrix;

            // Calculate interpolation ratio between bounding sections
            const auto alpha = static_cast<double>(refinement + 1) /
                               static_cast<double>(input.section_refinement + 1);

            // Interpolate grid location
            const auto grid_value = (1. - alpha) * left_location + alpha * section_location;

            // Interpolate mass and stiffness matrices from bounding sections
            auto mass_matrix = std::array<std::array<double, 6>, 6>{};
            auto stiffness_matrix = std::array<std::array<double, 6>, 6>{};
            for (auto component_1 = 0U; component_1 < 6; ++component_1) {
                for (auto component_2 = 0U; component_2 < 6; ++component_2) {
                    mass_matrix[component_1][component_2] =
                        (1. - alpha) * left_mass_matrix[component_1][component_2] +
                        alpha * section_mass_matrix[component_1][component_2];
                    stiffness_matrix[component_1][component_2] =
                        (1. - alpha) * left_stiffness_matrix[component_1][component_2] +
                        alpha * section_stiffness_matrix[component_1][component_2];
                }
            }

            // Calculate twist at current section location via linear interpolation
            twist = LinearInterp(section_location, input.ref_axis.twist_grid, input.ref_axis.twist);

            // Add refinement section
            q_twist = RotationVectorToQuaternion({twist * M_PI / 180., 0., 0.});
            sections.emplace_back(
                grid_value, RotateMatrix6(mass_matrix, q_twist),
                RotateMatrix6(stiffness_matrix, q_twist)
            );
        }

        // Add ending section
        twist = LinearInterp(section_location, input.ref_axis.twist_grid, input.ref_axis.twist);
        q_twist = RotationVectorToQuaternion({twist * M_PI / 180., 0., 0.});
        sections.emplace_back(
            section_location, RotateMatrix6(section_mass_matrix, q_twist),
            RotateMatrix6(section_stiffness_matrix, q_twist)
        );
    }

    return sections;
}
}  // namespace openturbine::interfaces::components
