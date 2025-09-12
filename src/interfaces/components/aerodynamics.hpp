#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#include "aerodynamics_input.hpp"
#include "interfaces/host_state.hpp"
#include "math/interpolation.hpp"
#include "math/quaternion_operations.hpp"
#include "math/vector_operations.hpp"
#include "model/node.hpp"

namespace openturbine::interfaces::components {

inline double CalculateAngleOfAttack(std::span<const double, 3> v_rel) {
    return std::atan2(-v_rel[2], v_rel[1]);
}

inline std::array<double, 6> CalculateAerodynamicLoad(
    std::span<double, 3> ref_axis_moment, std::span<const double, 3> v_inflow,
    std::span<const double, 3> v_motion, std::span<const double> aoa_polar,
    std::span<const double> cl_polar, std::span<const double> cd_polar,
    std::span<const double> cm_polar, double chord, double delta_s, double fluid_density,
    std::span<const double, 3> con_force, std::span<const double, 4> qqr
) {
    assert(aoa_polar.size() == cl_polar.size());
    assert(aoa_polar.size() == cd_polar.size());
    assert(aoa_polar.size() == cm_polar.size());

    auto v_rel_global = std::array<double, 3>{};
    for (auto direction = 0U; direction < 3U; ++direction) {
        v_rel_global[direction] = v_inflow[direction] - v_motion[direction];
    }

    const auto qqr_inv = math::QuaternionInverse(qqr);
    auto v_rel = math::RotateVectorByQuaternion(qqr_inv, v_rel_global);
    v_rel[0] = 0.;

    const auto velocity_magnitude = math::Norm(v_rel);

    const auto aoa = CalculateAngleOfAttack(v_rel);

    const auto polar_iterator =
        std::find_if(std::cbegin(aoa_polar), std::cend(aoa_polar), [aoa](auto polar) {
            return polar > aoa;
        });

    assert(polar_iterator != aoa_polar.end());

    const auto polar_index =
        static_cast<size_t>(std::distance(std::cbegin(aoa_polar), polar_iterator) - 1);

    const auto is_end = (polar_index == aoa_polar.size() - 1UL);

    const auto alpha = (is_end) ? 0.
                                : (aoa - aoa_polar[polar_index]) /
                                      (aoa_polar[polar_index + 1] - aoa_polar[polar_index]);

    const auto cl = (is_end)
                        ? cl_polar.back()
                        : (1. - alpha) * cl_polar[polar_index] + alpha * cl_polar[polar_index + 1];
    const auto cd = (is_end)
                        ? cd_polar.back()
                        : (1. - alpha) * cd_polar[polar_index] + alpha * cd_polar[polar_index + 1];
    const auto cm = (is_end)
                        ? cm_polar.back()
                        : (1. - alpha) * cm_polar[polar_index] + alpha * cm_polar[polar_index + 1];

    const auto dynamic_pressure = .5 * fluid_density * velocity_magnitude * velocity_magnitude;

    const auto drag_vector = std::array{
        v_rel[0] / velocity_magnitude, v_rel[1] / velocity_magnitude, v_rel[2] / velocity_magnitude
    };
    const auto lift_vector = math::CrossProduct(
        std::array{
            -1.,
            0.,
            0.,
        },
        drag_vector
    );

    auto force_local = std::array<double, 3>{};
    for (auto direction = 0U; direction < 3U; ++direction) {
        force_local[direction] = dynamic_pressure * chord * delta_s *
                                 (cl * lift_vector[direction] + cd * drag_vector[direction]);
    }
    const auto moment_local = std::array{cm * dynamic_pressure * chord * chord * delta_s, 0., 0.};

    const auto load_force = math::RotateVectorByQuaternion(qqr, force_local);
    const auto load_moment = math::RotateVectorByQuaternion(qqr, moment_local);

    const auto force_moment = math::CrossProduct(force_local, con_force);
    auto ref_axis_moment_local = std::array<double, 3>{};
    for (auto component = 0U; component < 3U; ++component) {
        ref_axis_moment_local[component] = force_moment[component] + moment_local[component];
    }

    const auto moment = math::RotateVectorByQuaternion(qqr, ref_axis_moment_local);
    std::ranges::copy(moment, std::begin(ref_axis_moment));
    //    ref_axis_moment = math::RotateVectorByQuaternion(qqr, ref_axis_moment_local);

    return {load_force[0],  load_force[1],  load_force[2],
            load_moment[0], load_moment[1], load_moment[2]};
}

inline std::array<double, 3> CalculateConMotionVector(
    double ac_to_ref_axis_horizontal, double chord_to_ref_axis_vertical
) {
    return {0., -ac_to_ref_axis_horizontal, chord_to_ref_axis_vertical};
}

inline std::vector<double> CalculateJacobianXi(std::span<const double> aero_node_xi) {
    const auto num_aero_nodes = aero_node_xi.size();
    const auto num_jacobian_nodes = 2 * num_aero_nodes + 1;

    auto jacobian_xi = std::vector<double>(num_jacobian_nodes);

    jacobian_xi[0] = aero_node_xi[0];
    jacobian_xi[1] = (3. * aero_node_xi[0] + aero_node_xi[1]) / 4.;

    for (auto i = 0U; i < num_aero_nodes - 1U; ++i) {
        jacobian_xi[2 * i + 2] = .5 * (aero_node_xi[i] + aero_node_xi[i + 1]);
        jacobian_xi[2 * i + 3] = aero_node_xi[i + 1];
    }

    jacobian_xi[num_jacobian_nodes - 2] =
        (aero_node_xi[num_aero_nodes - 2] + 3. * aero_node_xi[num_aero_nodes - 1]) / 4.;
    jacobian_xi[num_jacobian_nodes - 1] = aero_node_xi[num_aero_nodes - 1];

    return jacobian_xi;
}

inline std::vector<double> CalculateAeroNodeWidths(
    std::span<const double> jacobian_xi, std::span<const double> jacobian_integration_matrix,
    std::span<const double> node_x
) {
    const auto num_nodes = node_x.size() / 3;
    const auto num_jacobian_nodes = jacobian_xi.size();
    const auto num_aero_nodes = (num_jacobian_nodes - 1) / 2;

    auto tan = std::vector<double>(num_jacobian_nodes * 3);
    for (auto direction = 0U; direction < 3U; ++direction) {
        for (auto i = 0U; i < num_jacobian_nodes; ++i) {
            auto total = 0.;
            for (auto j = 0U; j < num_nodes; ++j) {
                total += jacobian_integration_matrix[i * num_nodes + j] *
                         node_x[direction * num_nodes + j];
            }
            tan[direction * num_jacobian_nodes + i] = total;
        }
    }

    auto j = std::vector<double>(num_jacobian_nodes);
    for (auto i = 0U; i < num_jacobian_nodes; ++i) {
        auto total = 0.;
        for (auto direction = 0U; direction < 3U; ++direction) {
            total +=
                tan[direction * num_jacobian_nodes + i] * tan[direction * num_jacobian_nodes + i];
        }
        j[i] = std::sqrt(total);
    }

    auto width = std::vector<double>(num_aero_nodes);
    for (auto i = 0U; i < num_aero_nodes; ++i) {
        const auto j0 = 2U * i;
        const auto j1 = j0 + 1U;
        const auto j2 = j1 + 1U;
        const auto h1 = jacobian_xi[j1] - jacobian_xi[j0];
        const auto h2 = jacobian_xi[j2] - jacobian_xi[j1];
        width[i] = ((h1 + h2) / 6.) *
                   ((2. - h2 / h1) * j[j0] + ((h1 + h2) * (h1 + h2) / (h1 * h2)) * j[j1] +
                    (2. - h1 / h2) * j[j2]);
    }
    return width;
}

class AerodynamicBody {
public:
    size_t id;

    std::vector<size_t> node_ids;
    std::vector<std::array<double, 7>> node_u;
    std::vector<std::array<double, 6>> node_v;
    std::vector<std::array<double, 6>> node_f;

    std::vector<std::array<double, 7>> xr_motion_map;
    std::vector<std::array<double, 7>> u_motion_map;
    std::vector<std::array<double, 6>> v_motion_map;
    std::vector<std::array<double, 4>> qqr_motion_map;
    std::vector<std::array<double, 3>> con_motion;
    std::vector<std::array<double, 3>> x_motion;
    std::vector<std::array<double, 3>> v_motion;

    std::vector<std::array<double, 3>> con_force;
    std::vector<std::array<double, 6>> loads;
    std::vector<std::array<double, 3>> ref_axis_moments;

    std::vector<double> jacobian_xi;
    std::vector<std::array<double, 3>> v_inflow;
    std::vector<std::array<double, 3>> v_rel;
    std::vector<double> twist;
    std::vector<double> chord;
    std::vector<double> delta_s;
    std::vector<size_t> polar_size;
    std::vector<std::vector<double>> aoa;
    std::vector<std::vector<double>> cl;
    std::vector<std::vector<double>> cd;
    std::vector<std::vector<double>> cm;

    std::vector<double> motion_interp;
    std::vector<double> shape_deriv_jac;

private:
    static std::vector<double> ExtractSectionXi(const AerodynamicBodyInput& input) {
        const auto n_sections = input.aero_sections.size();
        auto section_xi = std::vector<double>(n_sections);
        for (auto section = 0U; section < n_sections; ++section) {
            section_xi[section] = 2. * input.aero_sections[section].s - 1.;
        }
        return section_xi;
    }

    static std::vector<double> ExtractBeamNodeXi(
        const AerodynamicBodyInput& input, std::span<const Node> nodes
    ) {
        const auto n_nodes = input.beam_node_ids.size();
        auto beam_node_xi = std::vector<double>(n_nodes);
        for (auto node = 0U; node < n_nodes; ++node) {
            beam_node_xi[node] = 2. * nodes[input.beam_node_ids[node]].s - 1.;
        }
        return beam_node_xi;
    }

    static std::vector<double> ComputeMotionInterp(
        std::span<const double> section_xi, std::span<const double> beam_node_xi
    ) {
        const auto n_sections = section_xi.size();
        const auto n_nodes = beam_node_xi.size();
        auto interp = std::vector<double>(n_sections * n_nodes);
        interp.resize(n_sections * n_nodes);
        auto weights = std::vector<double>{};
        for (auto i = 0U; i < n_sections; ++i) {
            math::LagrangePolynomialInterpWeights(section_xi[i], beam_node_xi, weights);
            for (auto j = 0U; j < n_nodes; ++j) {
                interp[i * n_nodes + j] = weights[j];
            }
        }
        return interp;
    }

    static std::vector<std::array<double, 7>> ExtractNodeX(
        const AerodynamicBodyInput& input, std::span<const Node> nodes
    ) {
        const auto n_nodes = input.beam_node_ids.size();
        auto node_x = std::vector<std::array<double, 7>>(n_nodes);
        for (auto node = 0U; node < n_nodes; ++node) {
            for (auto component = 0U; component < 7U; ++component) {
                node_x[node][component] = nodes[input.beam_node_ids[node]].x0[component];
            }
        }
        return node_x;
    }

    static void InterpolateQuaternionFromNodesToSections(
        std::span<std::array<double, 7>> xr, std::span<const std::array<double, 7>> node_x,
        std::span<const double> interp
    ) {
        const auto n_nodes = node_x.size();
        const auto n_sections = xr.size();

        // Interpolate
        for (auto i = 0U; i < n_sections; ++i) {
            for (auto component = 0U; component < 7U; ++component) {
                xr[i][component] = 0.;
            }
            for (auto j = 0U; j < n_nodes; ++j) {
                const auto coeff = interp[i * n_nodes + j];
                for (auto component = 0U; component < 7U; ++component) {
                    xr[i][component] += coeff * node_x[j][component];
                }
            }
        }

        // Normalize
        for (auto section = 0U; section < n_sections; ++section) {
            const auto local_xr = xr[section];
            const auto length = std::sqrt(
                local_xr[3] * local_xr[3] + local_xr[4] * local_xr[4] + local_xr[5] * local_xr[5] +
                local_xr[6] * local_xr[6]
            );
            if (length > 1.e-16) {
                for (auto component = 3U; component < 7U; ++component) {
                    xr[section][component] /= length;
                }
            } else {
                xr[section][3] = 1.;
                for (auto component = 4U; component < 7U; ++component) {
                    xr[section][component] = 0.;
                }
            }
        }
    }

    static std::vector<std::array<double, 7>> InterpolateNodePositionsToSections(
        const AerodynamicBodyInput& input, std::span<const std::array<double, 7>> node_x,
        std::span<const double> interp, std::span<const double> section_xi,
        std::span<const double> beam_node_xi
    ) {
        const auto n_nodes = node_x.size();
        const auto n_sections = interp.size() / n_nodes;
        auto xr = std::vector<std::array<double, 7>>(n_sections);
        InterpolateQuaternionFromNodesToSections(xr, node_x, interp);

        // Calculate shape function derivative matrix to map from beam nodes to sections
        const auto shape_deriv_node = ComputeShapeDerivNode(section_xi, beam_node_xi);

        // Add Twist
        AddTwistToReferenceLocation(xr, node_x, input, shape_deriv_node);

        return xr;
    }

    static std::vector<double> ComputeShapeDerivNode(
        std::span<const double> section_xi, std::span<const double> beam_node_xi
    ) {
        const auto n_sections = section_xi.size();
        const auto n_nodes = beam_node_xi.size();
        auto shape_deriv_node = std::vector<double>(n_sections * n_nodes);
        auto weights = std::vector<double>{};
        for (auto i = 0U; i < n_sections; ++i) {
            math::LagrangePolynomialDerivWeights(section_xi[i], beam_node_xi, weights);
            for (auto j = 0U; j < n_nodes; ++j) {
                shape_deriv_node[i * n_nodes + j] = weights[j];
            }
        }
        return shape_deriv_node;
    }

    static void AddTwistToReferenceLocation(
        std::vector<std::array<double, 7>>& xr, std::span<const std::array<double, 7>> node_x,
        const AerodynamicBodyInput& input, std::span<const double> shape_deriv_node
    ) {
        const auto n_sections = xr.size();
        const auto n_nodes = node_x.size();

        auto x_tan = std::vector<std::array<double, 3>>(n_sections);
        for (auto i = 0U; i < n_sections; ++i) {
            for (auto component = 0U; component < 3U; ++component) {
                x_tan[i][component] = 0.;
            }
            for (auto j = 0U; j < n_nodes; ++j) {
                auto coeff = shape_deriv_node[i * n_nodes + j];
                for (auto component = 0U; component < 3U; ++component) {
                    x_tan[i][component] += coeff * node_x[j][component];
                }
            }
            const auto m = math::Norm(x_tan[i]);
            if (m > 1.e-16) {
                for (auto component = 0U; component < 3U; ++component) {
                    x_tan[i][component] /= m;
                }
            }
        }

        for (auto section = 0U; section < n_sections; ++section) {
            const auto q_twist =
                math::AxisAngleToQuaternion(x_tan[section], -input.aero_sections[section].twist);
            const auto qr =
                std::array{xr[section][3], xr[section][4], xr[section][5], xr[section][6]};
            auto q = math::QuaternionCompose(q_twist, qr);
            for (auto component = 3U; component < 7U; ++component) {
                xr[section][component] = q[component - 3U];
            }
        }
    }

    static std::vector<std::array<double, 3>> ComputeConMotion(const AerodynamicBodyInput& input) {
        const auto n_sections = input.aero_sections.size();
        auto con_motion = std::vector<std::array<double, 3>>(n_sections);
        for (auto section = 0U; section < n_sections; ++section) {
            const auto& node = input.aero_sections[section];
            const auto vec = CalculateConMotionVector(
                node.section_offset_y - node.aerodynamic_center, node.section_offset_x
            );
            for (auto component = 0U; component < 3U; ++component) {
                con_motion[section][component] = vec[component];
            }
        }
        return con_motion;
    }

    static std::vector<double> ComputeShapeDerivJacobian(
        std::span<const double> jacobian_xi, std::span<const double> beam_node_xi
    ) {
        const auto n_sections = jacobian_xi.size();
        const auto n_nodes = beam_node_xi.size();
        auto shape_deriv_jac = std::vector<double>(n_sections * n_nodes);
        auto weights = std::vector<double>{};
        for (auto i = 0U; i < n_sections; ++i) {
            math::LagrangePolynomialDerivWeights(jacobian_xi[i], beam_node_xi, weights);
            for (auto j = 0U; j < n_nodes; ++j) {
                shape_deriv_jac[i * n_nodes + j] = weights[j];
            }
        }
        return shape_deriv_jac;
    }

    static std::vector<double> ComputeDeltaS(
        std::span<const std::array<double, 7>> node_x, std::span<const double> jacobian_xi,
        std::span<const double> shape_deriv_jac
    ) {
        const auto n_sections = node_x.size();
        auto node_x_flat = std::vector<double>(3U * n_sections);
        for (auto direction = 0U; direction < 3U; ++direction) {
            for (auto section = 0U; section < n_sections; ++section) {
                node_x_flat[direction * n_sections + section] = node_x[section][direction];
            }
        }
        return CalculateAeroNodeWidths(jacobian_xi, shape_deriv_jac, node_x_flat);
    }

    template <typename T>
    static std::vector<std::vector<double>> ExtractPolar(size_t n_sections, T polar_extractor) {
        auto output = std::vector<std::vector<double>>(n_sections);
        for (auto section = 0U; section < n_sections; ++section) {
            const auto& polar_data = polar_extractor(section);
            const auto n_polar_points = polar_data.size();
            output[section].resize(n_polar_points);
            for (auto polar = 0U; polar < n_polar_points; ++polar) {
                output[section][polar] = polar_data[polar];
            }
        }
        return output;
    }

    static std::vector<std::array<double, 3>> InitializeConForce(
        std::span<const std::array<double, 3>> con_motion
    ) {
        const auto n_sections = con_motion.size();
        auto con_force = std::vector<std::array<double, 3>>(n_sections);
        for (auto section = 0U; section < n_sections; ++section) {
            for (auto component = 0U; component < 3U; ++component) {
                con_force[section][component] = -con_motion[section][component];
            }
        }
        return con_force;
    }

public:
    AerodynamicBody(const AerodynamicBodyInput& input, std::span<const Node> nodes)
        : id(input.id),
          node_ids(input.beam_node_ids),
          node_u(input.beam_node_ids.size()),
          node_v(input.beam_node_ids.size()),
          node_f(input.beam_node_ids.size()),
          u_motion_map(input.aero_sections.size()),
          v_motion_map(input.aero_sections.size()),
          qqr_motion_map(input.aero_sections.size()),
          x_motion(input.aero_sections.size()),
          v_motion(input.aero_sections.size()),
          loads(input.aero_sections.size()),
          ref_axis_moments(input.aero_sections.size()),
          v_inflow(input.aero_sections.size()) {
        // Get aerodynamic section location along beam
        const auto section_xi = ExtractSectionXi(input);

        // Get location of node along beam
        const auto beam_node_xi = ExtractBeamNodeXi(input, nodes);

        // Calculate shape function interpolation matrix to map from beam to aero points
        motion_interp = ComputeMotionInterp(section_xi, beam_node_xi);

        // Get node reference position
        const auto node_x = ExtractNodeX(input, nodes);

        // Interpolate node positions to aerodynamic sections on the beam reference axis
        xr_motion_map = InterpolateNodePositionsToSections(
            input, node_x, motion_interp, section_xi, beam_node_xi
        );

        // Calculate vector from beam reference to aerodynamic center
        con_motion = ComputeConMotion(input);

        jacobian_xi = CalculateJacobianXi(section_xi);

        // Calculate shae derivative matrix to calculate jacobians
        shape_deriv_jac = ComputeShapeDerivJacobian(jacobian_xi, beam_node_xi);

        // Calculate aero point widths
        delta_s = ComputeDeltaS(node_x, jacobian_xi, shape_deriv_jac);

        // Copy over the polar information
        const auto n_sections = input.aero_sections.size();
        twist = std::vector<double>(n_sections);
        std::ranges::transform(input.aero_sections, std::begin(twist), [](auto& section) {
            return section.twist;
        });
        chord = std::vector<double>(n_sections);
        std::ranges::transform(input.aero_sections, std::begin(chord), [](auto& section) {
            return section.chord;
        });

        aoa = ExtractPolar(n_sections, [&](size_t section) {
            return input.aero_sections[section].aoa;
        });
        cl = ExtractPolar(n_sections, [&](size_t section) {
            return input.aero_sections[section].cl;
        });
        cd = ExtractPolar(n_sections, [&](size_t section) {
            return input.aero_sections[section].cd;
        });
        cm = ExtractPolar(n_sections, [&](size_t section) {
            return input.aero_sections[section].cm;
        });

        con_force = InitializeConForce(con_motion);
    }

    template <typename DeviceType>
    void CalculateMotion(const HostState<DeviceType>& state) {
        // Copy beam node displacements from state
        for (auto node = 0U; node < node_u.size(); ++node) {
            for (auto component = 0U; component < 7U; ++component) {
                node_u[node][component] = state.q(node_ids[node], component);
            }
        }

        // Copy beam node velocities from state
        for (auto node = 0U; node < node_v.size(); ++node) {
            for (auto component = 0U; component < 6U; ++component) {
                node_v[node][component] = state.v(node_ids[node], component);
            }
        }

        InterpolateQuaternionFromNodesToSections(u_motion_map, node_u, motion_interp);

        // Interpolate beam node velocities to aerodynamic sections on the reference axis
        for (auto i = 0U; i < v_motion_map.size(); ++i) {
            for (auto component = 0U; component < 6U; ++component) {
                v_motion_map[i][component] = 0.;
            }
            for (auto j = 0U; j < node_v.size(); ++j) {
                const auto coeff = motion_interp[i * node_v.size() + j];
                for (auto component = 0U; component < 6U; ++component) {
                    v_motion_map[i][component] += coeff * node_v[j][component];
                }
            }
        }

        // Calculate global rotation of each section
        for (auto i = 0U; i < qqr_motion_map.size(); ++i) {
            const auto xr = std::array{
                xr_motion_map[i][3], xr_motion_map[i][4], xr_motion_map[i][5], xr_motion_map[i][6]
            };
            const auto u = std::array{
                u_motion_map[i][3], u_motion_map[i][4], u_motion_map[i][5], u_motion_map[i][6]
            };
            const auto qqr = math::QuaternionCompose(xr, u);
            for (auto component = 0U; component < 4U; ++component) {
                qqr_motion_map[i][component] = qqr[component];
            }
        }

        // Calculate motion of aerodynamic centers in global coordinates
        for (auto i = 0U; i < x_motion.size(); ++i) {
            const auto qqr_con = math::RotateVectorByQuaternion(qqr_motion_map[i], con_motion[i]);

            for (auto component = 0U; component < 3U; ++component) {
                x_motion[i][component] =
                    xr_motion_map[i][component] + u_motion_map[i][component] + qqr_con[component];
            }

            const auto omega =
                std::array{v_motion_map[i][3], v_motion_map[i][4], v_motion_map[i][5]};
            const auto omega_qqr_con = math::CrossProduct(omega, qqr_con);
            for (auto component = 0U; component < 3U; ++component) {
                v_motion[i][component] = v_motion_map[i][component] + omega_qqr_con[component];
            }
        }

        auto node_x = std::vector<double>(node_u.size() * 3U);
        for (auto node = 0U; node < node_u.size(); ++node) {
            for (auto component = 0U; component < 3U; ++component) {
                node_x[component * node_u.size() + node] = state.x(node_ids[node], component);
            }
        }

        delta_s = CalculateAeroNodeWidths(jacobian_xi, shape_deriv_jac, node_x);
    }

    void SetInflowFromVector(std::span<const std::array<double, 3>> inflow_velocity) {
        for (auto node = 0U; node < inflow_velocity.size(); ++node) {
            for (auto component = 0U; component < 3U; ++component) {
                v_inflow[node][component] = inflow_velocity[node][component];
            }
        }
    }

    template <typename T>
    void SetInflowFromFunction(const T& inflow_velocity_function) {
        for (auto node = 0U; node < v_inflow.size(); ++node) {
            const auto inflow_velocity = inflow_velocity_function(x_motion[node]);
            for (auto component = 0U; component < 3U; ++component) {
                v_inflow[node][component] = inflow_velocity[component];
            }
        }
    }

    void SetAerodynamicLoads(std::span<const std::array<double, 6>> aerodynamic_loads) {
        for (auto load = 0U; load < con_force.size(); ++load) {
            loads[load] = aerodynamic_loads[load];
            const auto rotated_con_force =
                math::RotateVectorByQuaternion(qqr_motion_map[load], con_force[load]);
            const auto new_load = std::array{loads[load][0], loads[load][1], loads[load][2]};
            const auto new_moment = std::array{loads[load][3], loads[load][4], loads[load][5]};
            const auto force_moment = math::CrossProduct(new_load, rotated_con_force);
            for (auto component = 0U; component < 3U; ++component) {
                ref_axis_moments[load][component] = force_moment[component] + new_moment[component];
            }
        }
    }

    void CalculateAerodynamicLoads(double fluid_density) {
        for (auto node = 0U; node < loads.size(); ++node) {
            const auto load = CalculateAerodynamicLoad(
                ref_axis_moments[node], v_inflow[node], v_motion[node], aoa[node], cl[node],
                cd[node], cm[node], chord[node], delta_s[node], fluid_density, con_force[node],
                qqr_motion_map[node]
            );
            for (auto component = 0U; component < 6U; ++component) {
                loads[node][component] = load[component];
            }
        }
    }

    void CalculateNodalLoads() {
        for (auto node = 0U; node < node_f.size(); ++node) {
            for (auto component = 0U; component < 3U; ++component) {
                node_f[node][component] = 0.;
            }
            for (auto section = 0U; section < loads.size(); ++section) {
                for (auto component = 0U; component < 3U; ++component) {
                    node_f[node][component] +=
                        motion_interp[section * node_f.size() + node] * loads[section][component];
                }
            }
        }

        for (auto node = 0U; node < node_f.size(); ++node) {
            for (auto component = 0U; component < 3U; ++component) {
                node_f[node][component + 3U] = 0.;
            }
            for (auto section = 0U; section < ref_axis_moments.size(); ++section) {
                for (auto component = 0U; component < 3U; ++component) {
                    node_f[node][component + 3U] += motion_interp[section * node_f.size() + node] *
                                                    ref_axis_moments[section][component];
                }
            }
        }
    }

    template <typename DeviceType>
    void AddNodalLoadsToState(HostState<DeviceType>& state) {
        for (auto node = 0U; node < node_f.size(); ++node) {
            for (auto component = 0U; component < 6U; ++component) {
                state.f(node_ids[node], component) += node_f[node][component];
            }
        }
    }
};

class Aerodynamics {
public:
    std::vector<AerodynamicBody> bodies;

    Aerodynamics(std::span<const AerodynamicBodyInput> inputs, std::span<const Node> nodes) {
        for (const auto& input : inputs) {
            bodies.emplace_back(input, nodes);
        }
    }

    template <typename DeviceType>
    void CalculateMotion(HostState<DeviceType>& state) {
        for (auto& body : bodies) {
            body.CalculateMotion(state);
        }
    }

    void SetInflowFromVector(
        std::span<const std::vector<std::array<double, 3>>> body_inflow_velocities
    ) {
        for (auto i = 0U; i < bodies.size(); ++i) {
            bodies[i].SetInflowFromVector(body_inflow_velocities[i]);
        }
    }

    template <typename T>
    void SetInflowFromFunction(const T& body_inflow_velocity_function) {
        for (auto& body : bodies) {
            body.SetInflowFromFunction(body_inflow_velocity_function);
        }
    }

    void SetAerodynamicLoads(std::span<const std::vector<std::array<double, 6>>> body_aero_loads) {
        for (auto i = 0U; i < bodies.size(); ++i) {
            bodies[i].SetAerodynamicLoads(body_aero_loads[i]);
        }
    }

    void CalculateAerodynamicLoads(double fluid_density) {
        for (auto& body : bodies) {
            body.CalculateAerodynamicLoads(fluid_density);
        }
    }

    void CalculateNodalLoads() {
        for (auto& body : bodies) {
            body.CalculateNodalLoads();
        }
    }

    template <typename DeviceType>
    void AddNodalLoadsToState(HostState<DeviceType>& state) {
        for (auto& body : bodies) {
            body.AddNodalLoadsToState(state);
        }
    }
};
}  // namespace openturbine::interfaces::components
