#include <initializer_list>

#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/restruct_poc/beams/beam_element.hpp"
#include "src/restruct_poc/beams/beam_node.hpp"
#include "src/restruct_poc/beams/beam_section.hpp"
#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/beams/beams_input.hpp"
#include "src/restruct_poc/beams/create_beams.hpp"
#include "src/restruct_poc/model/model.hpp"
#include "src/restruct_poc/state/copy_nodes_to_state.hpp"
#include "src/restruct_poc/state/state.hpp"
#include "src/restruct_poc/step/assemble_residual_vector.hpp"
#include "src/restruct_poc/step/update_system_variables.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine::tests {

inline auto SetUpBeams() {
    // Stiffness matrix for uniform composite beam section
    constexpr auto stiffness_matrix = std::array{
        std::array{1., 2., 3., 4., 5., 6.},      std::array{2., 4., 6., 8., 10., 12.},
        std::array{3., 6., 9., 12., 15., 18.},   std::array{4., 8., 12., 16., 20., 24.},
        std::array{5., 10., 15., 20., 25., 30.}, std::array{6., 12., 18., 24., 30., 36.},
    };

    // Mass matrix for uniform composite beam section
    constexpr auto mass_matrix = std::array{
        std::array{2., 0., 0., 0., 0.6, -0.4}, std::array{0., 2., 0., -0.6, 0., 0.2},
        std::array{0., 0., 2., 0.4, -0.2, 0.}, std::array{0., -0.6, 0.4, 1., 2., 3.},
        std::array{0.6, 0., -0.2, 2., 4., 6.}, std::array{-0.4, 0.2, 0., 3., 6., 9.},
    };

    auto model = Model();

    model.AddNode(
        {0, 0, 0, 0.9778215200524469, -0.01733607539094763, -0.09001900002195001,
         -0.18831121859148398},
        {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}
    );
    model.AddNode(
        {0.863365823230057, -0.2558982639254171, 0.11304112106827427, 0.9950113028068008,
         -0.002883848832932071, -0.030192109815745303, -0.09504013471947484},
        {0.002981602178886856, -0.00246675949494302, 0.003084570715675624, 0.9999627302042724,
         0.008633550973807708, 0, 0},
        {0.01726731646460114, -0.014285714285714285, 0.003084570715675624, 0.01726731646460114,
         -0.014285714285714285, 0.003084570715675624},
        {0.01726731646460114, -0.011304112106827427, 0.00606617289456248, 0.01726731646460114,
         -0.014285714285714285, -0.014285714285714285}
    );
    model.AddNode(
        {2.5, -0.25, 0, 0.9904718430204884, -0.009526411091536478, 0.09620741150793366,
         0.09807604012323785},
        {0.025, -0.0125, 0.027500000000000004, 0.9996875162757026, 0.02499739591471221, 0, 0},
        {0.05, -0.025, 0.027500000000000004, 0.05, -0.025, 0.027500000000000004},
        {0.05, 0, 0.052500000000000005, 0.05, -0.025, -0.025}
    );
    model.AddNode(
        {4.1366341767699435, 0.39875540678256005, -0.5416125496397031, 0.9472312341234699,
         -0.04969214162931507, 0.18127630174800594, 0.25965858850765167},
        {0.06844696924968459, -0.011818954790771264, 0.07977257214146725, 0.9991445348823055,
         0.04135454527402512, 0, 0},
        {0.08273268353539887, -0.01428571428571428, 0.07977257214146725, 0.08273268353539887,
         -0.01428571428571428, 0.07977257214146725},
        {0.08273268353539887, 0.05416125496397031, 0.14821954139115184, 0.08273268353539887,
         -0.01428571428571428, -0.01428571428571428}
    );
    model.AddNode(
        {5, 1, -1, 0.9210746582719719, -0.07193653093139739, 0.20507529985516368,
         0.32309554437664584},
        {0.1, 0, 0.12, 0.9987502603949663, 0.04997916927067825, 0, 0}, {0.1, 0, 0.12, 0.1, 0, 0.12},
        {0.1, 0.1, 0.22000000000000003, 0.1, 0, 0}
    );

    constexpr auto gravity = std::array{0., 0., 9.81};
    // Define beam initialization
    const auto beams_input = BeamsInput(
        {
            BeamElement(
                {
                    BeamNode(0., model.GetNode(0)),
                    BeamNode(0.1726731646460114, model.GetNode(1)),
                    BeamNode(0.5, model.GetNode(2)),
                    BeamNode(0.82732683535398865, model.GetNode(3)),
                    BeamNode(1., model.GetNode(4)),
                },
                {
                    BeamSection(0., mass_matrix, stiffness_matrix),
                    BeamSection(1., mass_matrix, stiffness_matrix),
                },
                BeamQuadrature{
                    {-0.9491079123427585, 0.1294849661688697},
                    {-0.7415311855993943, 0.27970539148927664},
                    {-0.40584515137739696, 0.3818300505051189},
                    {6.123233995736766e-17, 0.4179591836734694},
                    {0.4058451513773971, 0.3818300505051189},
                    {0.7415311855993945, 0.27970539148927664},
                    {0.9491079123427585, 0.1294849661688697},
                }
            ),
        },
        gravity
    );

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Create initial state
    auto parameters = StepParameters(false, 0, 0., 0.);
    State state(beams.num_nodes);
    CopyNodesToState(state, model.GetNodes());

    // Set the beam's initial state
    UpdateSystemVariables(parameters, beams, state);

    return beams;
}

TEST(BeamsTest, NodeInitialPositionX0) {
    const auto beams = SetUpBeams();
    expect_kokkos_view_2D_equal(
        Kokkos::subview(beams.node_x0, 0, Kokkos::ALL, Kokkos::ALL),
        {
            {0., 0., 0., 0.9778215200524469, -0.01733607539094763, -0.09001900002195001,
             -0.18831121859148398},
            {0.863365823230057, -0.2558982639254171, 0.11304112106827427, 0.9950113028068008,
             -0.002883848832932071, -0.030192109815745303, -0.09504013471947484},
            {2.5, -0.25, 0., 0.9904718430204884, -0.009526411091536478, 0.09620741150793366,
             0.09807604012323785},
            {4.1366341767699435, 0.39875540678256005, -0.5416125496397031, 0.9472312341234699,
             -0.04969214162931507, 0.18127630174800594, 0.25965858850765167},
            {5., 1., -1., 0.9210746582719719, -0.07193653093139739, 0.20507529985516368,
             0.32309554437664584},
        }
    );
}

TEST(BeamsTest, NodeInitialDisplacement) {
    const auto beams = SetUpBeams();
    expect_kokkos_view_2D_equal(
        Kokkos::subview(beams.node_u, 0, Kokkos::ALL, Kokkos::ALL),
        {
            {0., 0., 0., 1., 0., 0., 0.},
            {0.002981602178886856, -0.00246675949494302, 0.003084570715675624, 0.9999627302042724,
             0.008633550973807708, 0., 0.},
            {0.025, -0.0125, 0.027500000000000004, 0.9996875162757026, 0.02499739591471221, 0., 0.},
            {0.06844696924968459, -0.011818954790771264, 0.07977257214146725, 0.9991445348823055,
             0.04135454527402512, 0., 0.},
            {0.1, 0., 0.12, 0.9987502603949663, 0.04997916927067825, 0., 0.},
        }
    );
}

TEST(BeamsTest, NodeInitialVelocity) {
    const auto beams = SetUpBeams();
    expect_kokkos_view_2D_equal(
        Kokkos::subview(beams.node_u_dot, 0, Kokkos::ALL, Kokkos::ALL),
        {
            {0., 0., 0., 0., 0., 0},
            {0.01726731646460114, -0.014285714285714285, 0.003084570715675624, 0.01726731646460114,
             -0.014285714285714285, 0.003084570715675624},
            {0.05, -0.025, 0.027500000000000004, 0.05, -0.025, 0.027500000000000004},
            {0.08273268353539887, -0.01428571428571428, 0.07977257214146725, 0.08273268353539887,
             -0.01428571428571428, 0.07977257214146725},
            {0.1, 0., 0.12, 0.1, 0., 0.12},
        }
    );
}

TEST(BeamsTest, NodeInitialAcceleration) {
    const auto beams = SetUpBeams();
    expect_kokkos_view_2D_equal(
        Kokkos::subview(beams.node_u_ddot, 0, Kokkos::ALL, Kokkos::ALL),
        {
            {0., 0., 0., 0., 0., 0},
            {0.01726731646460114, -0.011304112106827427, 0.00606617289456248, 0.01726731646460114,
             -0.014285714285714285, -0.014285714285714285},
            {0.05, 0., 0.052500000000000005, 0.05, -0.025, -0.025},
            {0.08273268353539887, 0.05416125496397031, 0.14821954139115184, 0.08273268353539887,
             -0.01428571428571428, -0.01428571428571428},
            {0.1, 0.1, 0.22000000000000003, 0.1, 0., 0.},
        }
    );
}

TEST(BeamsTest, QuadraturePointMassMatrixInMaterialFrame) {
    const auto beams = SetUpBeams();
    auto Mstar = View_NxN("Mstar", beams.qp_Mstar.extent(2), beams.qp_Mstar.extent(3));
    Kokkos::deep_copy(Mstar, Kokkos::subview(beams.qp_Mstar, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        Mstar,
        {
            {2., 0., 0., 0., 0.6, -0.4},
            {0., 2., 0., -0.6, 0., 0.2},
            {0., 0., 2., 0.4, -0.2, 0.},
            {0., -0.6, 0.4, 1., 2., 3.},
            {0.6, 0., -0.2, 2., 4., 6.},
            {-0.4, 0.2, 0., 3., 6., 9.},

        }
    );
}

TEST(BeamsTest, QuadraturePointStiffnessMatrixInMaterialFrame) {
    const auto beams = SetUpBeams();
    auto Cstar = View_NxN("Cstar", beams.qp_Cstar.extent(2), beams.qp_Cstar.extent(3));
    Kokkos::deep_copy(Cstar, Kokkos::subview(beams.qp_Cstar, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        Cstar,
        {
            {1., 2., 3., 4., 5., 6.},
            {2., 4., 6., 8., 10., 12.},
            {3., 6., 9., 12., 15., 18.},
            {4., 8., 12., 16., 20., 24.},
            {5., 10., 15., 20., 25., 30.},
            {6., 12., 18., 24., 30., 36.},
        }
    );
}

TEST(BeamsTest, QuadraturePointRR0) {
    const auto beams = SetUpBeams();
    auto RR0 = View_NxN("RR0", beams.qp_RR0.extent(2), beams.qp_RR0.extent(3));
    Kokkos::deep_copy(RR0, Kokkos::subview(beams.qp_RR0, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        RR0,
        {
            {0.9246873610951006, 0.34700636042507577, -0.156652066872805, 0.0, 0.0, 0.0},
            {-0.3426571011111718, 0.937858102036658, 0.05484789423748749, 0.0, 0.0, 0.0},
            {0.16594997827377847, 0.002960788533623304, 0.9861297269843315, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.9246873610951006, 0.34700636042507577, -0.156652066872805},
            {0.0, 0.0, 0.0, -0.3426571011111718, 0.937858102036658, 0.05484789423748749},
            {0.0, 0.0, 0.0, 0.16594997827377847, 0.002960788533623304, 0.9861297269843315},
        }
    );
}

TEST(BeamsTest, QuadraturePointMassMatrixInGlobalFrame) {
    const auto beams = SetUpBeams();
    auto Muu = View_NxN("Muu", beams.qp_Muu.extent(2), beams.qp_Muu.extent(3));
    Kokkos::deep_copy(Muu, Kokkos::subview(beams.qp_Muu, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        Muu,
        {
            {2.000000000000001, 5.204170427930421e-17, -5.551115123125783e-17,
             -4.163336342344337e-17, 0.626052147258804, -0.3395205571349214},
            {5.204170427930421e-17, 2.0000000000000018, 1.3877787807814457e-17, -0.6260521472588039,
             -3.469446951953614e-18, 0.22974877626536766},
            {-5.551115123125783e-17, 1.3877787807814457e-17, 2.0000000000000013, 0.33952055713492146,
             -0.22974877626536772, -1.3877787807814457e-17},
            {4.163336342344337e-17, -0.626052147258804, 0.3395205571349214, 1.3196125048858467,
             1.9501108129670985, 3.5958678677753957},
            {0.6260521472588039, 3.469446951953614e-18, -0.22974877626536766, 1.9501108129670985,
             2.881855217930184, 5.313939345820573},
            {-0.33952055713492146, 0.22974877626536772, 1.3877787807814457e-17, 3.5958678677753957,
             5.3139393458205735, 9.79853227718398},
        }
    );
}

TEST(BeamsTest, QuadraturePointStiffnessMatrixInGlobalFrame) {
    const auto beams = SetUpBeams();
    auto Cuu = View_NxN("Cuu", beams.qp_Cuu.extent(2), beams.qp_Cuu.extent(3));
    Kokkos::deep_copy(Cuu, Kokkos::subview(beams.qp_Cuu, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        Cuu,
        {
            {1.3196125048858467, 1.9501108129670985, 3.5958678677753957, 5.1623043394880055,
             4.190329885612304, 7.576404967559343},
            {1.9501108129670985, 2.881855217930184, 5.313939345820573, 7.628804270184899,
             6.192429663690275, 11.196339225304031},
            {3.5958678677753957, 5.3139393458205735, 9.79853227718398, 14.066981200400345,
             11.418406945420463, 20.64526599682174},
            {5.162304339488006, 7.628804270184899, 14.066981200400342, 20.194857198478893,
             16.392507703808057, 29.638782670624547},
            {4.190329885612305, 6.192429663690274, 11.418406945420463, 16.392507703808064,
             13.3060762043738, 24.058301996624227},
            {7.576404967559343, 11.196339225304033, 20.64526599682174, 29.63878267062455,
             24.058301996624223, 43.499066597147355},
        }
    );
}

TEST(BeamsTest, QuadraturePointMatrixOuu) {
    const auto beams = SetUpBeams();
    auto Ouu = View_NxN("Ouu", beams.qp_Ouu.extent(2), beams.qp_Ouu.extent(3));
    Kokkos::deep_copy(Ouu, Kokkos::subview(beams.qp_Ouu, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        Ouu,
        {
            {0., 0., 0., 1.558035187754702, 3.3878498808227704, -2.4090666622503774},
            {0., 0., 0., 2.023578567654382, 4.594419401889352, -3.2342585893237827},
            {0., 0., 0., 4.396793221398987, 8.369447695979858, -6.152454589644055},
            {0., 0., 0., 6.095010301161761, 12.749853070301084, -9.15756802064953},
            {0., 0., 0., 4.359848751597227, 9.872327664027363, -6.769213486026294},
            {0., 0., 0., 9.270255102567306, 17.449495035002304, -12.963070176574703},
        }
    );
}

TEST(BeamsTest, QuadraturePointMatrixPuu) {
    const auto beams = SetUpBeams();
    auto Puu = View_NxN("Puu", beams.qp_Puu.extent(2), beams.qp_Puu.extent(3));
    Kokkos::deep_copy(Puu, Kokkos::subview(beams.qp_Puu, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        Puu,
        {
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {1.558035187754702, 2.0235785676543823, 4.396793221398987, 6.095010301161762,
             4.947423115431053, 8.945281658389643},
            {3.3878498808227704, 4.594419401889353, 8.369447695979858, 12.162278706467262,
             9.872327664027365, 17.849848197376673},
            {-2.4090666622503774, -3.2342585893237827, -6.152454589644055, -8.832594576471866,
             -7.169566648400663, -12.963070176574703},
        }
    );
}

TEST(BeamsTest, QuadraturePointMatrixQuu) {
    const auto beams = SetUpBeams();
    auto Quu = View_NxN("Quu", beams.qp_Quu.extent(2), beams.qp_Quu.extent(3));
    Kokkos::deep_copy(Quu, Kokkos::subview(beams.qp_Quu, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        Quu,
        {
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 0., 0., 0.},
            {0., 0., 0., 1.8447536136896558, 3.635630275656869, -2.64866290970237},
            {0., 0., 0., 3.8107321412411204, 7.1833246136382, -5.294123557503821},
            {0., 0., 0., -2.4075516854952794, -5.414954154362819, 3.820161491912928},
        }
    );
}

TEST(BeamsTest, QuadraturePointMatrixGuu) {
    const auto beams = SetUpBeams();
    auto Guu = View_NxN("Guu", beams.qp_Guu.extent(2), beams.qp_Guu.extent(3));
    Kokkos::deep_copy(Guu, Kokkos::subview(beams.qp_Guu, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        Guu,
        {
            {0., 0., 0., -0.0008012182534494841, 0.002003432464632351, 0.0015631511018243545},
            {0., 0., 0., -0.002297634478952118, 0.0006253629923483924, -0.0015967098417843995},
            {0., 0., 0., -0.0031711581076346268, 0.0031271320551441812, -0.0002573417597137699},
            {0., 0., 0., -0.009044140792420115, -0.016755394335528064, -0.022806270184157214},
            {0., 0., 0., -0.005674132164451402, -0.013394960837037522, -0.025943451454082944},
            {0., 0., 0., 0.006396216051163168, 0.013413253109011812, 0.022439101629457635},
        }
    );
}

TEST(BeamsTest, QuadraturePointMatrixKuu) {
    const auto beams = SetUpBeams();
    auto Kuu = View_NxN("Kuu", beams.qp_Kuu.extent(2), beams.qp_Kuu.extent(3));
    Kokkos::deep_copy(Kuu, Kokkos::subview(beams.qp_Kuu, 0, 0, Kokkos::ALL, Kokkos::ALL));
    expect_kokkos_view_2D_equal(
        Kuu,
        {
            {0., 0., 0., -0.0023904728226588536, 0.0005658527664274542, 0.0005703830914904407},
            {0., 0., 0., -0.0008599439459226316, -0.000971811812092634, 0.0008426153626567674},
            {0., 0., 0., -0.0015972403418206974, 0.0015555222717217175, -0.000257435063678694},
            {0., 0., 0., 0.004762288305421506, -0.016524233223710137, 0.007213755243428677},
            {0., 0., 0., 0.035164381478288514, 0.017626317482204206, -0.022463736936512112},
            {0., 0., 0., -0.0025828596476940593, 0.04278211835291491, -0.022253736971069835},
        }
    );
}

}  // namespace openturbine::tests
