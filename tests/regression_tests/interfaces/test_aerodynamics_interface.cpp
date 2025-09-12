#include <array>
#include <numbers>
#include <ranges>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "elements/beams/hollow_circle_properties.hpp"
#include "interfaces/components/aerodynamics.hpp"
#include "interfaces/components/inflow.hpp"
#include "interfaces/turbine/turbine_interface.hpp"
#include "interfaces/turbine/turbine_interface_builder.hpp"

namespace openturbine::tests {

TEST(AerodynamicsInterfaceTest, IEA15_Turbine) {
    constexpr auto time_step = 0.01;
    constexpr auto duration = 1.;
    constexpr auto n_blades = 3U;
    constexpr auto n_blade_nodes = 11U;
    constexpr auto n_tower_nodes = 11U;
    constexpr auto n_steps = static_cast<unsigned>(duration / time_step);
    constexpr auto write_output = false;

    auto builder = interfaces::TurbineInterfaceBuilder{};
    builder.Solution()
        .EnableDynamicSolve()
        .SetTimeStep(time_step)
        .SetDampingFactor(0.0)
        .SetGravity({0., 0., -9.81})
        .SetMaximumNonlinearIterations(6)
        .SetAbsoluteErrorTolerance(1e-6)
        .SetRelativeErrorTolerance(1e-4);

    if (write_output) {
        builder.Solution().SetOutputFile("TurbineInterfaceTest.IEA15");
    }

    const YAML::Node wio = YAML::LoadFile("interfaces_test_files/IEA-15-240-RWT-aero.yaml");

    auto& turbine_builder = builder.Turbine();

    for (auto blade : std::views::iota(0U, n_blades)) {
        auto& blade_builder = turbine_builder.Blade(blade)
                                  .SetElementOrder(n_blade_nodes - 1)
                                  .PrescribedRootMotion(false);

        const auto& wio_blade = wio["components"]["blade"];
        const auto& ref_axis = wio_blade["reference_axis"];
        const auto& blade_twist = wio_blade["outer_shape"]["twist"];
        const auto& inertia_matrix = wio_blade["structure"]["elastic_properties"]["inertia_matrix"];
        const auto& stiffness_matrix =
            wio_blade["structure"]["elastic_properties"]["stiffness_matrix"];

        const auto axis_grid = ref_axis["x"]["grid"].as<std::vector<double>>();
        const auto x_values = ref_axis["x"]["values"].as<std::vector<double>>();
        const auto y_values = ref_axis["y"]["values"].as<std::vector<double>>();
        const auto z_values = ref_axis["z"]["values"].as<std::vector<double>>();
        for (auto i : std::views::iota(0U, axis_grid.size())) {
            blade_builder.AddRefAxisPoint(
                axis_grid[i], {x_values[i], y_values[i], z_values[i]},
                interfaces::components::ReferenceAxisOrientation::Z
            );
        }

        const auto twist_grid = blade_twist["grid"].as<std::vector<double>>();
        const auto twist_values = blade_twist["values"].as<std::vector<double>>();
        for (auto i : std::views::iota(0U, twist_grid.size())) {
            blade_builder.AddRefAxisTwist(twist_grid[i], twist_values[i] * std::numbers::pi / 180.);
        }

        const auto k_grid = stiffness_matrix["grid"].as<std::vector<double>>();
        const auto m_grid = inertia_matrix["grid"].as<std::vector<double>>();
        const auto mass = inertia_matrix["mass"].as<std::vector<double>>();
        const auto cm_x = inertia_matrix["cm_x"].as<std::vector<double>>();
        const auto cm_y = inertia_matrix["cm_y"].as<std::vector<double>>();
        const auto i_cp = inertia_matrix["i_cp"].as<std::vector<double>>();
        const auto i_edge = inertia_matrix["i_edge"].as<std::vector<double>>();
        const auto i_flap = inertia_matrix["i_flap"].as<std::vector<double>>();
        const auto i_plr = inertia_matrix["i_plr"].as<std::vector<double>>();

        const auto K11 = stiffness_matrix["K11"].as<std::vector<double>>();
        const auto K12 = stiffness_matrix["K12"].as<std::vector<double>>();
        const auto K13 = stiffness_matrix["K13"].as<std::vector<double>>();
        const auto K14 = stiffness_matrix["K14"].as<std::vector<double>>();
        const auto K15 = stiffness_matrix["K15"].as<std::vector<double>>();
        const auto K16 = stiffness_matrix["K16"].as<std::vector<double>>();
        const auto K22 = stiffness_matrix["K22"].as<std::vector<double>>();
        const auto K23 = stiffness_matrix["K23"].as<std::vector<double>>();
        const auto K24 = stiffness_matrix["K24"].as<std::vector<double>>();
        const auto K25 = stiffness_matrix["K25"].as<std::vector<double>>();
        const auto K26 = stiffness_matrix["K26"].as<std::vector<double>>();
        const auto K33 = stiffness_matrix["K33"].as<std::vector<double>>();
        const auto K34 = stiffness_matrix["K34"].as<std::vector<double>>();
        const auto K35 = stiffness_matrix["K35"].as<std::vector<double>>();
        const auto K36 = stiffness_matrix["K36"].as<std::vector<double>>();
        const auto K44 = stiffness_matrix["K44"].as<std::vector<double>>();
        const auto K45 = stiffness_matrix["K45"].as<std::vector<double>>();
        const auto K46 = stiffness_matrix["K46"].as<std::vector<double>>();
        const auto K55 = stiffness_matrix["K55"].as<std::vector<double>>();
        const auto K56 = stiffness_matrix["K56"].as<std::vector<double>>();
        const auto K66 = stiffness_matrix["K66"].as<std::vector<double>>();
        const auto n_sections = k_grid.size();

        for (auto section : std::views::iota(0U, n_sections)) {
            if (abs(m_grid[section] - k_grid[section]) > 1e-8) {
                throw std::runtime_error("stiffness and mass matrices not on same grid");
            }
            blade_builder.AddSection(
                m_grid[section],
                {{{mass[section], 0., 0., 0., 0., -mass[section] * cm_y[section]},
                  {0., mass[section], 0., 0., 0., mass[section] * cm_x[section]},
                  {0., 0., mass[section], mass[section] * cm_y[section],
                   -mass[section] * cm_x[section], 0.},
                  {0., 0., mass[section] * cm_y[section], i_edge[section], -i_cp[section], 0.},
                  {0., 0., -mass[section] * cm_x[section], -i_cp[section], i_flap[section], 0.},
                  {-mass[section] * cm_y[section], mass[section] * cm_x[section], 0., 0., 0.,
                   i_plr[section]}}},
                {{
                    {K11[section], K12[section], K13[section], K14[section], K15[section],
                     K16[section]},
                    {K12[section], K22[section], K23[section], K24[section], K25[section],
                     K26[section]},
                    {K13[section], K23[section], K33[section], K34[section], K35[section],
                     K36[section]},
                    {K14[section], K24[section], K34[section], K44[section], K45[section],
                     K46[section]},
                    {K15[section], K25[section], K35[section], K45[section], K55[section],
                     K56[section]},
                    {K16[section], K26[section], K36[section], K46[section], K56[section],
                     K66[section]},
                }},
                interfaces::components::ReferenceAxisOrientation::Z
            );
        }
    }

    auto& tower_builder =
        turbine_builder.Tower().SetElementOrder(n_tower_nodes - 1).PrescribedRootMotion(false);

    auto& wio_tower = wio["components"]["tower"];
    auto& wio_tower_diameter = wio_tower["outer_shape"]["outer_diameter"];
    const auto tower_diameter_grid = wio_tower_diameter["grid"].as<std::vector<double>>();
    const auto tower_diameter_values = wio_tower_diameter["values"].as<std::vector<double>>();
    const auto tower_wall_thickness =
        wio_tower["structure"]["layers"][0]["thickness"]["values"].as<std::vector<double>>();
    const auto tower_material_name =
        wio_tower["structure"]["layers"][0]["material"].as<std::string>();
    auto& tower_ref_axis = wio_tower["reference_axis"];

    const auto axis_grid = tower_ref_axis["x"]["grid"].as<std::vector<double>>();
    const auto x_values = tower_ref_axis["x"]["values"].as<std::vector<double>>();
    const auto y_values = tower_ref_axis["y"]["values"].as<std::vector<double>>();
    const auto z_values = tower_ref_axis["z"]["values"].as<std::vector<double>>();
    for (auto i : std::views::iota(0U, axis_grid.size())) {
        tower_builder.AddRefAxisPoint(
            axis_grid[i], {x_values[i], y_values[i], z_values[i]},
            interfaces::components::ReferenceAxisOrientation::Z
        );
    }

    const auto tower_base_position =
        std::array<double, 7>{x_values[0], y_values[0], z_values[0], 1., 0., 0., 0.};
    turbine_builder.SetTowerBasePosition(tower_base_position);
    tower_builder.AddRefAxisTwist(0., 0.).AddRefAxisTwist(1., 0.);

    auto tower_material = std::find_if(
                              std::begin(wio["materials"]), std::end(wio["materials"]),
                              [&tower_material_name](const YAML::Node& node) {
                                  return node["name"].as<std::string>() == tower_material_name;
                              }
    )->as<YAML::Node>();

    auto elastic_modulus = tower_material["E"].as<double>();
    auto shear_modulus = tower_material["G"].as<double>();
    auto poisson_ratio = tower_material["nu"].as<double>();
    auto density = tower_material["rho"].as<double>();

    for (auto i : std::views::iota(0U, tower_diameter_grid.size())) {
        const auto section = beams::GenerateHollowCircleSection(
            tower_diameter_grid[i], elastic_modulus, shear_modulus, density,
            tower_diameter_values[i], tower_wall_thickness[i], poisson_ratio
        );

        tower_builder.AddSection(
            tower_diameter_grid[i], section.M_star, section.C_star,
            interfaces::components::ReferenceAxisOrientation::Z
        );
    }

    auto& wio_hub = wio["components"]["hub"];
    turbine_builder.SetAzimuthAngle(0.)
        .SetConeAngle(wio_hub["cone_angle"].as<double>() * std::numbers::pi / 180.)
        .SetHubDiameter(wio_hub["diameter"].as<double>())
        .SetRotorApexToHub(0.);

    auto& wio_drivetrain = wio["components"]["drivetrain"];
    turbine_builder.SetTowerAxisToRotorApex(wio_drivetrain["outer_shape"]["overhang"].as<double>())
        .SetTowerTopToRotorApex(wio_drivetrain["outer_shape"]["distance_tt_hub"].as<double>())
        .SetShaftTiltAngle(
            wio_drivetrain["outer_shape"]["uptilt"].as<double>() * std::numbers::pi / 180.
        );

    const auto drivetrain_mass = wio_drivetrain["elastic_properties"]["mass"].as<double>();
    const auto drivetrain_inertia =
        wio_drivetrain["elastic_properties"]["inertia_tt"].as<std::vector<double>>();

    turbine_builder
        .SetYawBearingInertiaMatrix(std::array{
            std::array{drivetrain_mass, 0., 0., 0., 0., 0.},
            std::array{0., drivetrain_mass, 0., 0., 0., 0.},
            std::array{0., 0., drivetrain_mass, 0., 0., 0.},
            std::array{
                0., 0., 0., drivetrain_inertia[0], drivetrain_inertia[3], drivetrain_inertia[4]
            },
            std::array{
                0., 0., 0., drivetrain_inertia[3], drivetrain_inertia[1], drivetrain_inertia[5]
            },
            std::array{
                0., 0., 0., drivetrain_inertia[4], drivetrain_inertia[5], drivetrain_inertia[2]
            }
        })
        .SetHubInertiaMatrix(std::array{
            std::array{69131., 0., 0., 0., 0., 0.}, std::array{0., 69131., 0., 0., 0., 0.},
            std::array{0., 0., 69131., 0., 0., 0.},
            std::array{0., 0., 0., 969952. + 1836784., 0., 0.}, std::array{0., 0., 0., 0., 1., 0.},
            std::array{0., 0., 0., 0., 0., 1.}
        });

    auto& aero_builder =
        builder.Aerodynamics().EnableAero().SetNumberOfAirfoils(1UL).SetAirfoilToBladeMap(
            std::array{0UL, 0UL, 0UL}
        );

    {
        auto& airfoil_io = wio["airfoils"];
        auto aero_sections = std::vector<interfaces::components::AerodynamicSection>{};
        auto id = 0UL;
        for (auto& af : airfoil_io) {
            const auto s = af["spanwise_position"].as<double>();
            const auto chord = af["chord"].as<double>();
            const auto twist = af["twist"].as<double>() * std::numbers::pi / 180.;
            const auto section_offset_x = af["section_offset_x"].as<double>();
            const auto section_offset_y = af["section_offset_y"].as<double>();
            const auto aerodynamic_center = af["aerodynamic_center"].as<double>();
            auto aoa = af["polars"][0]["re_sets"][0]["cl"]["grid"].as<std::vector<double>>();
            std::ranges::transform(aoa, std::begin(aoa), [](auto degrees) {
                return degrees * std::numbers::pi / 180.;
            });
            const auto cl = af["polars"][0]["re_sets"][0]["cl"]["values"].as<std::vector<double>>();
            const auto cd = af["polars"][0]["re_sets"][0]["cd"]["values"].as<std::vector<double>>();
            const auto cm = af["polars"][0]["re_sets"][0]["cm"]["values"].as<std::vector<double>>();

            aero_sections.emplace_back(
                id, s, chord, section_offset_x, section_offset_y, aerodynamic_center, twist, aoa, cl,
                cd, cm
            );
            ++id;
        }

        aero_builder.SetAirfoilSections(0UL, aero_sections);
    }

    auto turbine_interface = builder.Build();

    constexpr auto fluid_density = 1.225;
    constexpr auto vel_h = 10.6;
    constexpr auto h_ref = 150.;
    constexpr auto pl_exp = 0.12;
    constexpr auto flow_angle = 0.;
    auto inflow = interfaces::components::Inflow::SteadyWind(vel_h, h_ref, pl_exp, flow_angle);

    for (auto i : std::views::iota(1U, n_steps)) {
        const auto t = i * time_step;

        turbine_interface.UpdateAerodynamicLoads(
            fluid_density,
            [t, &inflow](const std::array<double, 3>& pos) {
                return inflow.Velocity(t, pos);
            }
        );

        const auto converged = turbine_interface.Step();
        ASSERT_EQ(converged, true);
        if (i == 100) {
            EXPECT_NEAR(turbine_interface.CalculateAzimuthAngle(), 6.28289, 1.e-5);
            EXPECT_NEAR(turbine_interface.CalculateRotorSpeed(), -0.000707775, 1.e-9);
        }
    }
}
}  // namespace openturbine::tests
