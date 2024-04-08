#include <fstream>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

#include "src/restruct_poc/beams.hpp"
#include "src/restruct_poc/solver.hpp"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::restruct_poc::tests {

Quaternion CRV2Quat(double c1, double c2, double c3) {
    double c0 = 2. - (c1 * c1 + c2 * c2 + c3 * c3) / 8.;
    double e0 = c0 / (2. - c0);
    return Quaternion(e0, c1 / (4 - c0), c2 / (4 - c0), c3 / (4 - c0));
}

TEST(RotatingBeamTest, IEA15Rotor) {
    // Gravity vector
    std::array<double, 3> gravity = {-9.81, 0., 0.};

    // Rotor angular velocity in rad/s
    Vector omega(0., 0., 0.104719755);  // 7.55 rpm

    // Solution parameters
    const bool is_dynamic_solve(true);
    const size_t max_iter(6);
    const double step_size(0.01);  // seconds
    const double rho_inf(0.0);
    const double t_end(0.1);
    const size_t num_steps(t_end / step_size + 1.0);

    // Element quadrature
    BeamQuadrature trapz_quadrature{
        {-1.0, 0.005},  {-0.99, 0.01},   {-0.98, 0.01},  {-0.97, 0.01},   {-0.96, 0.01},
        {-0.95, 0.01},  {-0.94, 0.01},   {-0.93, 0.01},  {-0.92, 0.01},   {-0.91, 0.01},
        {-0.9, 0.0175}, {-0.875, 0.025}, {-0.85, 0.025}, {-0.825, 0.025}, {-0.8, 0.0375},
        {-0.75, 0.05},  {-0.7, 0.05},    {-0.65, 0.05},  {-0.6, 0.05},    {-0.55, 0.05},
        {-0.5, 0.05},   {-0.45, 0.05},   {-0.4, 0.05},   {-0.35, 0.05},   {-0.3, 0.05},
        {-0.25, 0.05},  {-0.2, 0.05},    {-0.15, 0.05},  {-0.1, 0.05},    {-0.05, 0.05},
        {0, 0.05},      {0.05, 0.05},    {0.1, 0.05},    {0.15, 0.05},    {0.2, 0.05},
        {0.25, 0.05},   {0.3, 0.05},     {0.35, 0.05},   {0.4, 0.05},     {0.45, 0.05},
        {0.5, 0.05},    {0.55, 0.05},    {0.6, 0.05},    {0.65, 0.05},    {0.7, 0.05},
        {0.75, 0.05},   {0.8, 0.05},     {0.85, 0.05},   {0.9, 0.05},     {0.95, 0.05},
        {1.0, 0.025},
    };

    // Node location from [-1, 1]
    std::vector<double> node_xi = {
        -1, -0.93400143040805916, -0.78448347366314441, -0.56523532699620493, -0.29575813558693936,
        0,  0.29575813558693936,  0.56523532699620493,  0.78448347366314441,  0.93400143040805916,
        1};

    // Node location [0, 1]
    std::vector<double> node_loc;
    for (const auto& xi : node_xi) {
        node_loc.push_back((xi + 1.) / 2.);
    }

    auto quat_fix = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(M_PI, 0., 0.) *
                    openturbine::gen_alpha_solver::quaternion_from_rotation_vector(0, M_PI_2, 0);

    // Node Coordinates
    std::vector<Vector> node_coords = {
        quat_fix * Vector(0, 0, 0),
        quat_fix * Vector(0.054696641731336287, 0, 3.8609163211285589),
        quat_fix * Vector(0.1464345809622267, 0, 12.607716790706057),
        quat_fix * Vector(0.23918824799640687, 0, 25.433733370721974),
        quat_fix * Vector(0.24301416634939574, 0, 41.198149068164057),
        quat_fix * Vector(-0.054037571255419574, 0, 58.500000000000043),
        quat_fix * Vector(-0.78540604521129809, 0, 75.801850931835958),
        quat_fix * Vector(-1.8044719127555222, 0, 91.566266629277948),
        quat_fix * Vector(-2.8241493860343772, 0, 104.39228320929396),
        quat_fix * Vector(-3.6162361535819612, 0, 113.13908367887151),
        quat_fix * Vector(-4, 0, 117),
    };

    // Node Rotation
    std::vector<Quaternion> node_rotation = {
        quat_fix * CRV2Quat(-0.0021734394298078691, 0.015872057390738433, -0.27259360477300898),
        quat_fix * CRV2Quat(-0.0017042282820729485, 0.012577816397165636, -0.26975435179131918),
        quat_fix * CRV2Quat(-0.0010177987013682073, 0.0089015782174543031, -0.22793456173578383),
        quat_fix * CRV2Quat(-0.00037213843223208627, 0.0051655366806687237, -0.14389838670749208),
        quat_fix * CRV2Quat(0.00022604382590256085, -0.0063747247346252461, -0.070896313300515726),
        quat_fix * CRV2Quat(0.00045998405477728344, -0.029256250711078839, -0.031441555906115012),
        quat_fix * CRV2Quat(6.4369468929472103e-06, -0.054818776535083324, -0.00023480049351383748),
        quat_fix * CRV2Quat(-0.0010526555705805531, -0.073276452450789661, 0.028719948501210704),
        quat_fix * CRV2Quat(-0.0016040356615223935, -0.085406812116008785, 0.037541802521014724),
        quat_fix * CRV2Quat(-0.0013917564635221034, -0.095949232909243246, 0.02899204787144392),
        quat_fix * CRV2Quat(-0.0011112659013357538, -0.10249334150582726, 0.021669770494811492),
    };

    std::vector<BeamSection> material_sections = {
        BeamSection{
            0.0,
            {{
                {3127.4021155424143, 0.0, 0.0, 0.0, -0.23227931035313248, -73.93195471060494},
                {0.0, 3127.4021155424143, 0.0, 0.23227931035313248, 0.0, 0.0},
                {0.0, 0.0, 3127.4021155424143, 73.93195471060494, 0.0, 0.0},
                {0.0, 0.23227931035313248, 73.93195471060494, 20334.260749419092, 0.0, 0.0},
                {-0.23227931035313248, 0.0, 0.0, 0.0, 10166.284427210068, -1.0683130734106165},
                {-73.93195471060494, 0.0, 0.0, 0.0, -1.0683130734106165, 10167.976322208995},
            }},
            {{
                {46051081603.60474, 0.0, 0.0, 0.0, -18829097.30285156, -1092494742.1422234},
                {0.0, 6729088765.395921, -2653738.5828939164, -38985021.66520424, 0.0, 0.0},
                {0.0, -2653738.5919261174, 6740375994.200792, 148446683.0081474, 0.0, 0.0},
                {0.0, -38985021.66540279, 148446683.00808364, 87489183048.03288, 0.0, 0.0},
                {-18829097.300796624, 0.0, 0.0, 0.0, 149729095916.4146, 22581466.167482525},
                {-1092494742.1446135, 0.0, 0.0, 0.0, 22581466.165237263, 149629012637.96594},
            }},
        },
        BeamSection{
            0.01,
            {{
                {2964.7325318133635, 0.0, 0.0, 0.0, -0.1696455253166024, 39.689341530367614},
                {0.0, 2964.7325318133635, 0.0, 0.1696455253166024, 0.0, 0.0},
                {0.0, 0.0, 2964.7325318133635, -39.689341530367614, 0.0, 0.0},
                {0.0, 0.1696455253166024, -39.689341530367614, 19348.32414628797, 0.0, 0.0},
                {-0.1696455253166024, 0.0, 0.0, 0.0, 9677.506658278195, 0.9133955577605082},
                {39.689341530367614, 0.0, 0.0, 0.0, 0.9133955577605082, 9670.817488009772},
            }},
            {{
                {43751344063.19816, 0.0, 0.0, 0.0, -3336008.299362885, 582609637.8397082},
                {0.0, 6388480690.310746, 376400.64745506924, -16608264.93565775, 0.0, 0.0},
                {0.0, 376400.6474385026, 6384592445.510168, -93807158.54721366, 0.0, 0.0},
                {0.0, -16608264.935654417, -93807158.54720719, 83285587642.82162, 0.0, 0.0},
                {-3336008.2982923416, 0.0, 0.0, 0.0, 143137826352.7965, 232559270.06063834},
                {582609637.8406111, 0.0, 0.0, 0.0, 232559270.06092423, 142328864112.50156},
            }},
        },
        BeamSection{
            0.02,
            {{
                {2805.1273323852683, 0.0, 0.0, 0.0, 2.405715769683996, 140.75148902158406},
                {0.0, 2805.1273323852683, 0.0, -2.405715769683996, 0.0, 0.0},
                {0.0, 0.0, 2805.1273323852683, -140.75148902158406, 0.0, 0.0},
                {0.0, -2.405715769683996, -140.75148902158406, 18373.702908261756, 0.0, 0.0},
                {2.405715769683996, 0.0, 0.0, 0.0, 9189.979166446146, 6.143364484361837},
                {140.75148902158406, 0.0, 0.0, 0.0, 6.143364484361837, 9183.723741815673},
            }},
            {{
                {41569662567.47037, 0.0, 0.0, 0.0, 39669072.53786528, 2071574619.5397494},
                {0.0, 6055417688.515367, 5351360.127523829, -5594870.114677965, 0.0, 0.0},
                {0.0, 5351360.127132061, 6033428473.028316, -302258358.414727, 0.0, 0.0},
                {0.0, -5594870.114658471, -302258358.41472834, 79106159653.8364, 0.0, 0.0},
                {39669072.53784892, 0.0, 0.0, 0.0, 137008541712.70963, 606773042.400328},
                {2071574619.5395474, 0.0, 0.0, 0.0, 606773042.4005727, 135212838311.38753},
            }},
        },
        BeamSection{
            0.03,
            {{
                {2591.1367835087917, 0.0, 0.0, 0.0, 8.850322233785226, 186.50710374919134},
                {0.0, 2591.1367835087917, 0.0, -8.850322233785226, 0.0, 0.0},
                {0.0, 0.0, 2591.1367835087917, -186.50710374919134, 0.0, 0.0},
                {0.0, -8.850322233785226, -186.50710374919134, 16280.22568568902, 0.0, 0.0},
                {8.850322233785226, 0.0, 0.0, 0.0, 7913.55513775584, 30.1760176639648},
                {186.50710374919134, 0.0, 0.0, 0.0, 30.1760176639648, 8366.670547933158},
            }},
            {{
                {38669982721.7398, 0.0, 0.0, 0.0, 139140684.30100253, 2739604966.9813576},
                {0.0, 5758216767.868562, -2545105.3698775643, -19709135.853905857, 0.0, 0.0},
                {0.0, -2545105.3701622505, 5399937528.5606785, -376430490.9109713, 0.0, 0.0},
                {0.0, -19709135.853861183, -376430490.91097593, 69973323042.1117, 0.0, 0.0},
                {139140684.2999871, 0.0, 0.0, 0.0, 119514275124.89417, 1348337968.6187627},
                {2739604966.980646, 0.0, 0.0, 0.0, 1348337968.6178887, 123284345388.39743},
            }},
        },
        BeamSection{
            0.04,
            {{
                {2389.1154059162695, 0.0, 0.0, 0.0, 12.729809187017585, 225.67253185892582},
                {0.0, 2389.1154059162695, 0.0, -12.729809187017585, 0.0, 0.0},
                {0.0, 0.0, 2389.1154059162695, -225.67253185892582, 0.0, 0.0},
                {0.0, -12.729809187017585, -225.67253185892582, 14384.707079520182, 0.0, 0.0},
                {12.729809187017585, 0.0, 0.0, 0.0, 6778.707874270462, 51.13814800535637},
                {225.67253185892582, 0.0, 0.0, 0.0, 51.13814800535637, 7605.999205249705},
            }},
            {{
                {36038990116.73735, 0.0, 0.0, 0.0, 201959858.66306534, 3310499851.237116},
                {0.0, 5468722459.955154, -6445969.845172357, -28462348.93837689, 0.0, 0.0},
                {0.0, -6445969.845348487, 4800672070.892391, -410026336.0979204, 0.0, 0.0},
                {0.0, -28462348.93834525, -410026336.0979162, 61533150488.50107, 0.0, 0.0},
                {201959858.66272968, 0.0, 0.0, 0.0, 104365370041.97543, 2131730557.8970866},
                {3310499851.237596, 0.0, 0.0, 0.0, 2131730557.8957243, 112223397786.61401},
            }},
        },
        BeamSection{
            0.05,
            {{
                {2203.52246956623, 0.0, 0.0, 0.0, 15.77615497545956, 266.66982245842087},
                {0.0, 2203.52246956623, 0.0, -15.77615497545956, 0.0, 0.0},
                {0.0, 0.0, 2203.52246956623, -266.66982245842087, 0.0, 0.0},
                {0.0, -15.77615497545956, -266.66982245842087, 12763.75220428584, 0.0, 0.0},
                {15.77615497545956, 0.0, 0.0, 0.0, 5816.730513104689, 69.34293642402783},
                {266.66982245842087, 0.0, 0.0, 0.0, 69.34293642402783, 6947.021691181144},
            }},
            {{
                {33746441410.19616, 0.0, 0.0, 0.0, 248900545.30290735, 3921047165.837773},
                {0.0, 5195984070.294925, -6455163.70291466, -35814075.943787806, 0.0, 0.0},
                {0.0, -6455163.702384659, 4245595846.60386, -423187047.4596364, 0.0, 0.0},
                {0.0, -35814075.94381699, -423187047.45961386, 54135840079.7236, 0.0, 0.0},
                {248900545.3026905, 0.0, 0.0, 0.0, 91978034767.20616, 2938603122.297642},
                {3921047165.838074, 0.0, 0.0, 0.0, 2938603122.299237, 102686583994.11134},
            }},
        },
        BeamSection{
            0.075,
            {{
                {1793.5621800535478, 0.0, 0.0, 0.0, 21.32062998431816, 349.63893753405364},
                {0.0, 1793.5621800535478, 0.0, -21.32062998431816, 0.0, 0.0},
                {0.0, 0.0, 1793.5621800535478, -349.63893753405364, 0.0, 0.0},
                {0.0, -21.32062998431816, -349.63893753405364, 9456.81881736644, 0.0, 0.0},
                {21.32062998431816, 0.0, 0.0, 0.0, 3914.596828171232, 103.28185672319495},
                {349.63893753405364, 0.0, 0.0, 0.0, 103.28185672319495, 5542.22198919525},
            }},
            {{
                {29212071034.730743, 0.0, 0.0, 0.0, 316621483.96654403, 5225114500.366909},
                {0.0, 4552086385.17254, 6056529.37060037, -53346766.10196565, 0.0, 0.0},
                {0.0, 6056529.370011804, 3015180949.0747685, -335581833.17485225, 0.0, 0.0},
                {0.0, -53346766.101946875, -335581833.174896, 38418612295.439545, 0.0, 0.0},
                {316621483.966785, 0.0, 0.0, 0.0, 68686121530.687454, 4823615805.89942},
                {5225114500.366529, 0.0, 0.0, 0.0, 4823615805.899531, 82456226297.96957},
            }},
        },
        BeamSection{
            0.1,
            {{
                {1694.5071226473444, 0.0, 0.0, 0.0, 33.63149776511567, 525.4310002430254},
                {0.0, 1694.5071226473444, 0.0, -33.63149776511567, 0.0, 0.0},
                {0.0, 0.0, 1694.5071226473444, -525.4310002430254, 0.0, 0.0},
                {0.0, -33.63149776511567, -525.4310002430254, 7669.567333210458, 0.0, 0.0},
                {33.63149776511567, 0.0, 0.0, 0.0, 2785.9557474336307, 159.42251092667695},
                {525.4310002430254, 0.0, 0.0, 0.0, 159.42251092667695, 4883.611585776837},
            }},
            {{
                {27339201435.297394, 0.0, 0.0, 0.0, 518317006.6590065, 8519455804.023233},
                {0.0, 3985354344.8681536, -113285786.00307085, -87781196.91790846, 0.0, 0.0},
                {0.0, -113285786.00360973, 2419253237.9808803, -235382612.4995677, 0.0, 0.0},
                {0.0, -87781196.91766477, -235382612.4997524, 25428346793.25849, 0.0, 0.0},
                {518317006.6611653, 0.0, 0.0, 0.0, 52263394041.374664, 6133557676.831906},
                {8519455804.023159, 0.0, 0.0, 0.0, 6133557676.831683, 72781368868.39719},
            }},
        },
        BeamSection{
            0.15,
            {{
                {1050.5041617289855, 0.0, 0.0, 0.0, 24.488449094119666, 610.1711896797851},
                {0.0, 1050.5041617289855, 0.0, -24.488449094119666, 0.0, 0.0},
                {0.0, 0.0, 1050.5041617289855, -610.1711896797851, 0.0, 0.0},
                {0.0, -24.488449094119666, -610.1711896797851, 4277.195390327287, 0.0, 0.0},
                {24.488449094119666, 0.0, 0.0, 0.0, 956.3088698304845, 109.41235764515795},
                {610.1711896797851, 0.0, 0.0, 0.0, 109.41235764515795, 3320.886520496825},
            }},
            {{
                {22665764106.85165, 0.0, 0.0, 0.0, 252339646.012082, 10699908369.669474},
                {0.0, 2314391966.661147, -80833117.64911054, -102518319.262035, 0.0, 0.0},
                {0.0, -80833117.672499, 936314786.8968203, 110222775.82612513, 0.0, 0.0},
                {0.0, -102518319.26448892, 110222775.82439752, 8076640558.8585005, 0.0, 0.0},
                {252339646.31990957, 0.0, 0.0, 0.0, 27988353518.859665, 4839685559.73392},
                {10699908369.111029, 0.0, 0.0, 0.0, 4839685559.488184, 54125167812.039314},
            }},
        },
        BeamSection{
            0.2,
            {{
                {668.8389205216013, 0.0, 0.0, 0.0, 20.275144559753706, 519.3241440864526},
                {0.0, 668.8389205216013, 0.0, -20.275144559753706, 0.0, 0.0},
                {0.0, 0.0, 668.8389205216013, -519.3241440864526, 0.0, 0.0},
                {0.0, -20.275144559753706, -519.3241440864526, 2729.9223355815425, 0.0, 0.0},
                {20.275144559753706, 0.0, 0.0, 0.0, 476.36366817186706, 74.25657502591736},
                {519.3241440864526, 0.0, 0.0, 0.0, 74.25657502591736, 2253.558667409682},
            }},
            {{
                {20450235425.96102, 0.0, 0.0, 0.0, 318201927.73381317, 9176487472.148022},
                {0.0, 963021463.3416004, -50723215.96503455, -63418511.854790874, 0.0, 0.0},
                {0.0, -50723215.96362021, 459476847.22473943, 44624062.11459313, 0.0, 0.0},
                {0.0, -63418511.85317413, 44624062.11472014, 2768066330.973254, 0.0, 0.0},
                {318201927.72660166, 0.0, 0.0, 0.0, 21820483410.15772, 3395447917.857598},
                {9176487472.167862, 0.0, 0.0, 0.0, 3395447917.8743534, 39196915729.85904},
            }},
        },
        BeamSection{
            0.25,
            {{
                {531.0612749513001, 0.0, 0.0, 0.0, 18.40686653041935, 469.18731596545734},
                {0.0, 531.0612749513001, 0.0, -18.40686653041935, 0.0, 0.0},
                {0.0, 0.0, 531.0612749513001, -469.18731596545734, 0.0, 0.0},
                {0.0, -18.40686653041935, -469.18731596545734, 2117.1378877835145, 0.0, 0.0},
                {18.40686653041935, 0.0, 0.0, 0.0, 268.55950272107657, 51.00047866730059},
                {469.18731596545734, 0.0, 0.0, 0.0, 51.00047866730059, 1848.5783850624332},
            }},
            {{
                {20473859852.028965, 0.0, 0.0, 0.0, 370375609.629183, 9004633626.168808},
                {0.0, 480065126.665447, -19062499.890920684, -35649958.00142443, 0.0, 0.0},
                {0.0, -19062499.890864983, 281350049.44872, 39788481.558390185, 0.0, 0.0},
                {0.0, -35649958.00112997, 39788481.55812293, 1026128471.8607168, 0.0, 0.0},
                {370375609.6294636, 0.0, 0.0, 0.0, 15615256841.106361, 1524671060.6535838},
                {9004633626.168045, 0.0, 0.0, 0.0, 1524671060.6524186, 35299428705.26732},
            }},
        },
        BeamSection{
            0.3,
            {{
                {483.98515809475083, 0.0, 0.0, 0.0, 15.24938260415129, 429.5473703129163},
                {0.0, 483.98515809475083, 0.0, -15.24938260415129, 0.0, 0.0},
                {0.0, 0.0, 483.98515809475083, -429.5473703129163, 0.0, 0.0},
                {0.0, -15.24938260415129, -429.5473703129163, 1768.8037116010007, 0.0, 0.0},
                {15.24938260415129, 0.0, 0.0, 0.0, 204.06774305708595, 34.07084223757462},
                {429.5473703129163, 0.0, 0.0, 0.0, 34.07084223757462, 1564.7359685439142},
            }},
            {{
                {21366371296.473003, 0.0, 0.0, 0.0, 349173302.5918751, 8306288511.852894},
                {0.0, 346901262.38535905, -4177613.499178857, -26944490.484182574, 0.0, 0.0},
                {0.0, -4177613.4992410876, 224644322.01692382, 22822651.106314287, 0.0, 0.0},
                {0.0, -26944490.48435339, 22822651.106341563, 627053086.269001, 0.0, 0.0},
                {349173302.5921183, 0.0, 0.0, 0.0, 13362814423.387623, 615674444.315585},
                {8306288511.853354, 0.0, 0.0, 0.0, 615674444.313989, 31051165577.557346},
            }},
        },
        BeamSection{
            0.35,
            {{
                {458.0838867335805, 0.0, 0.0, 0.0, 12.541536147390817, 387.99790618434287},
                {0.0, 458.0838867335805, 0.0, -12.541536147390817, 0.0, 0.0},
                {0.0, 0.0, 458.0838867335805, -387.99790618434287, 0.0, 0.0},
                {0.0, -12.541536147390817, -387.99790618434287, 1441.4647526558301, 0.0, 0.0},
                {12.541536147390817, 0.0, 0.0, 0.0, 154.74717074687635, 23.38641910057473},
                {387.99790618434287, 0.0, 0.0, 0.0, 23.38641910057473, 1286.7175819089484},
            }},
            {{
                {21785402237.03918, 0.0, 0.0, 0.0, 281941707.05108696, 7885934214.420893},
                {0.0, 324078740.9094225, 2993741.656463931, -22977520.67687867, 0.0, 0.0},
                {0.0, 2993741.656823819, 187672692.57290956, 14817252.655003047, 0.0, 0.0},
                {0.0, -22977520.67701866, 14817252.655248243, 461211585.49078345, 0.0, 0.0},
                {281941707.0508255, 0.0, 0.0, 0.0, 10663481378.13921, 91553134.31278107},
                {7885934214.419314, 0.0, 0.0, 0.0, 91553134.31292474, 26088284004.54947},
            }},
        },
        BeamSection{
            0.4,
            {{
                {433.8716017058155, 0.0, 0.0, 0.0, 9.385100666405265, 345.8892017251126},
                {0.0, 433.8716017058155, 0.0, -9.385100666405265, 0.0, 0.0},
                {0.0, 0.0, 433.8716017058155, -345.8892017251126, 0.0, 0.0},
                {0.0, -9.385100666405265, -345.8892017251126, 1196.3062966511723, 0.0, 0.0},
                {9.385100666405265, 0.0, 0.0, 0.0, 120.1630106896849, 18.75152324183863},
                {345.8892017251126, 0.0, 0.0, 0.0, 18.75152324183863, 1076.1432859614897},
            }},
            {{
                {21621041012.04207, 0.0, 0.0, 0.0, 151464382.0873222, 6912268845.2773905},
                {0.0, 317531903.30316025, 3309906.6224801578, -19104038.48691781, 0.0, 0.0},
                {0.0, 3309906.622644328, 161609316.90177184, 14196051.945945216, 0.0, 0.0},
                {0.0, -19104038.48709074, 14196051.945912855, 364794590.1572093, 0.0, 0.0},
                {151464382.08704397, 0.0, 0.0, 0.0, 8537806429.681531, 10375272.853540858},
                {6912268845.277576, 0.0, 0.0, 0.0, 10375272.853503438, 22217425410.95908},
            }},
        },
        BeamSection{
            0.45,
            {{
                {404.9178066535816, 0.0, 0.0, 0.0, 7.475763615859694, 311.07188808957676},
                {0.0, 404.9178066535816, 0.0, -7.475763615859694, 0.0, 0.0},
                {0.0, 0.0, 404.9178066535816, -311.07188808957676, 0.0, 0.0},
                {0.0, -7.475763615859694, -311.07188808957676, 986.2598738903342, 0.0, 0.0},
                {7.475763615859694, 0.0, 0.0, 0.0, 90.50849177530833, 15.603626309920882},
                {311.07188808957676, 0.0, 0.0, 0.0, 15.603626309920882, 895.7513821150329},
            }},
            {{
                {20779461548.755814, 0.0, 0.0, 0.0, 81958836.22604564, 6380607368.417731},
                {0.0, 309152872.5553294, 2612569.618519041, -15276680.601082614, 0.0, 0.0},
                {0.0, 2612569.6184869376, 137969183.99018735, 10110737.414443506, 0.0, 0.0},
                {0.0, -15276680.601433454, 10110737.414507402, 285660364.09887934, 0.0, 0.0},
                {81958836.22572584, 0.0, 0.0, 0.0, 6549534664.533833, -2259343.505242591},
                {6380607368.418019, 0.0, 0.0, 0.0, -2259343.50512368, 18717308525.793583},
            }},
        },
        BeamSection{
            0.5,
            {{
                {377.73123303078404, 0.0, 0.0, 0.0, 7.8155616062122695, 272.8500883723055},
                {0.0, 377.73123303078404, 0.0, -7.8155616062122695, 0.0, 0.0},
                {0.0, 0.0, 377.73123303078404, -272.8500883723055, 0.0, 0.0},
                {0.0, -7.8155616062122695, -272.8500883723055, 801.9237987714802, 0.0, 0.0},
                {7.8155616062122695, 0.0, 0.0, 0.0, 66.51587194758397, 13.644216263894043},
                {272.8500883723055, 0.0, 0.0, 0.0, 13.644216263894043, 735.407926823897},
            }},
            {{
                {19923519424.805744, 0.0, 0.0, 0.0, 145792800.15431416, 5510503208.482732},
                {0.0, 303218881.82122993, 1443459.8263998972, -12426129.212282114, 0.0, 0.0},
                {0.0, 1443459.826543158, 116600478.4523859, 9756447.07571729, 0.0, 0.0},
                {0.0, -12426129.21235434, 9756447.07565128, 220544333.37495336, 0.0, 0.0},
                {145792800.1543057, 0.0, 0.0, 0.0, 4893346064.323091, 28845801.840368368},
                {5510503208.483071, 0.0, 0.0, 0.0, 28845801.840308778, 15586807919.093891},
            }},
        },
        BeamSection{
            0.55,
            {{
                {350.48434003585845, 0.0, 0.0, 0.0, 8.182976209383485, 237.97949220138435},
                {0.0, 350.48434003585845, 0.0, -8.182976209383485, 0.0, 0.0},
                {0.0, 0.0, 350.48434003585845, -237.97949220138435, 0.0, 0.0},
                {0.0, -8.182976209383485, -237.97949220138435, 641.4957168880029, 0.0, 0.0},
                {8.182976209383485, 0.0, 0.0, 0.0, 48.3879168461972, 11.662250592755035},
                {237.97949220138435, 0.0, 0.0, 0.0, 11.662250592755035, 593.1078000418055},
            }},
            {{
                {18925122694.713356, 0.0, 0.0, 0.0, 215770838.70957416, 4939813431.154279},
                {0.0, 292909738.469172, 997091.0378785358, -11590547.025402421, 0.0, 0.0},
                {0.0, 997091.0381108759, 97422515.98334335, 7650615.789120105, 0.0, 0.0},
                {0.0, -11590547.025477929, 7650615.789076826, 168037150.44173867, 0.0, 0.0},
                {215770838.7094618, 0.0, 0.0, 0.0, 3604017503.6991744, 43922352.51911654},
                {4939813431.154144, 0.0, 0.0, 0.0, 43922352.51904449, 12739863625.745161},
            }},
        },
        BeamSection{
            0.6,
            {{
                {307.3478250743852, 0.0, 0.0, 0.0, 7.95168602738849, 164.7291992520334},
                {0.0, 307.3478250743852, 0.0, -7.95168602738849, 0.0, 0.0},
                {0.0, 0.0, 307.3478250743852, -164.7291992520334, 0.0, 0.0},
                {0.0, -7.95168602738849, -164.7291992520334, 424.3318450020865, 0.0, 0.0},
                {7.95168602738849, 0.0, 0.0, 0.0, 35.22217015645534, 7.836966520296293},
                {164.7291992520334, 0.0, 0.0, 0.0, 7.836966520296293, 389.10967484563156},
            }},
            {{
                {17539655765.235256, 0.0, 0.0, 0.0, 299940988.574154, 3340368066.1970086},
                {0.0, 300285135.273037, -122590.70891244167, -12779793.365596209, 0.0, 0.0},
                {0.0, -122590.70890719072, 81358328.95851956, 7709138.634627286, 0.0, 0.0},
                {0.0, -12779793.36563295, 7709138.63463516, 128226839.00696014, 0.0, 0.0},
                {299940988.5740754, 0.0, 0.0, 0.0, 2655217532.05773, 29747619.28784807},
                {3340368066.1971393, 0.0, 0.0, 0.0, 29747619.287767652, 8379809772.736287},
            }},
        },
        BeamSection{
            0.65,
            {{
                {260.2653356372791, 0.0, 0.0, 0.0, 7.292661316519065, 93.43468152003294},
                {0.0, 260.2653356372791, 0.0, -7.292661316519065, 0.0, 0.0},
                {0.0, 0.0, 260.2653356372791, -93.43468152003294, 0.0, 0.0},
                {0.0, -7.292661316519065, -93.43468152003294, 240.68491551038218, 0.0, 0.0},
                {7.292661316519065, 0.0, 0.0, 0.0, 25.215546621059765, 4.176133069084764},
                {93.43468152003294, 0.0, 0.0, 0.0, 4.176133069084764, 215.46936888932268},
            }},
            {{
                {15853197990.946262, 0.0, 0.0, 0.0, 358354291.37689996, 1772039285.1553001},
                {0.0, 334604810.69196564, -1150059.0012533022, -15336313.590033066, 0.0, 0.0},
                {0.0, -1150059.0011940901, 67080158.82424255, 8354890.671745444, 0.0, 0.0},
                {0.0, -15336313.590018397, 8354890.671770222, 95377930.58201036, 0.0, 0.0},
                {358354291.37691766, 0.0, 0.0, 0.0, 1917970315.4393296, 2424421.8977263183},
                {1772039285.1548736, 0.0, 0.0, 0.0, 2424421.8976960694, 4637021573.468788},
            }},
        },
        BeamSection{
            0.7,
            {{
                {223.94256194555769, 0.0, 0.0, 0.0, 7.223037382444947, 57.93504767403098},
                {0.0, 223.94256194555769, 0.0, -7.223037382444947, 0.0, 0.0},
                {0.0, 0.0, 223.94256194555769, -57.93504767403098, 0.0, 0.0},
                {0.0, -7.223037382444947, -57.93504767403098, 149.39987156305426, 0.0, 0.0},
                {7.223037382444947, 0.0, 0.0, 0.0, 17.745330077956677, 2.6421037604616724},
                {57.93504767403098, 0.0, 0.0, 0.0, 2.6421037604616724, 131.65454148509653},
            }},
            {{
                {14061229868.525568, 0.0, 0.0, 0.0, 411634704.48295814, 1013471013.85025},
                {0.0, 376257910.10373855, -1941394.5822713473, -18041654.778164484, 0.0, 0.0},
                {0.0, -1941394.5823109567, 55553240.58366735, 8124277.187599225, 0.0, 0.0},
                {0.0, -18041654.777826957, 8124277.187591464, 71456369.55600888, 0.0, 0.0},
                {411634704.484275, 0.0, 0.0, 0.0, 1351951128.6929338, 12258851.352306124},
                {1013471013.8422865, 0.0, 0.0, 0.0, 12258851.347575188, 2889083609.8505793},
            }},
        },
        BeamSection{
            0.75,
            {{
                {179.58407067190126, 0.0, 0.0, 0.0, 6.572357722853001, 38.0523975504382},
                {0.0, 179.58407067190126, 0.0, -6.572357722853001, 0.0, 0.0},
                {0.0, 0.0, 179.58407067190126, -38.0523975504382, 0.0, 0.0},
                {0.0, -6.572357722853001, -38.0523975504382, 95.92081642240376, 0.0, 0.0},
                {6.572357722853001, 0.0, 0.0, 0.0, 11.271194923324614, 1.7996118776795522},
                {38.0523975504382, 0.0, 0.0, 0.0, 1.7996118776795522, 84.64962149907886},
            }},
            {{
                {11003024089.793188, 0.0, 0.0, 0.0, 386594144.7272241, 625226575.7818947},
                {0.0, 395913126.7681885, -2180662.5761840697, -19527620.335504133, 0.0, 0.0},
                {0.0, -2180662.5761736603, 44994384.56866734, 7061108.353385276, 0.0, 0.0},
                {0.0, -19527620.335547354, 7061108.3533984525, 52519301.176315226, 0.0, 0.0},
                {386594144.7271621, 0.0, 0.0, 0.0, 838106592.74152, 15695344.266543552},
                {625226575.7819192, 0.0, 0.0, 0.0, 15695344.266583264, 1877272361.936434},
            }},
        },
        BeamSection{
            0.8,
            {{
                {129.3080054809469, 0.0, 0.0, 0.0, 4.71916228979406, 23.226025130813625},
                {0.0, 129.3080054809469, 0.0, -4.71916228979406, 0.0, 0.0},
                {0.0, 0.0, 129.3080054809469, -23.226025130813625, 0.0, 0.0},
                {0.0, -4.71916228979406, -23.226025130813625, 58.13938290669439, 0.0, 0.0},
                {4.71916228979406, 0.0, 0.0, 0.0, 6.684580575763557, 1.0405277605856154},
                {23.226025130813625, 0.0, 0.0, 0.0, 1.0405277605856154, 51.45480233093064},
            }},
            {{
                {7310260724.863415, 0.0, 0.0, 0.0, 261309728.4735774, 354179115.2888132},
                {0.0, 375210610.3003366, -2151842.3141690264, -17825049.8174219, 0.0, 0.0},
                {0.0, -2151842.3143516458, 36509915.3832926, 6124281.751202558, 0.0, 0.0},
                {0.0, -17825049.817279924, 6124281.751209516, 38106482.51374478, 0.0, 0.0},
                {261309728.47355214, 0.0, 0.0, 0.0, 466233432.88131785, 6469059.311190724},
                {354179115.2887366, 0.0, 0.0, 0.0, 6469059.311161065, 1099701983.6863494},
            }},
        },
        BeamSection{
            0.85,
            {{
                {84.45573745266202, 0.0, 0.0, 0.0, 2.8210572700220213, 16.39546225863475},
                {0.0, 84.45573745266202, 0.0, -2.8210572700220213, 0.0, 0.0},
                {0.0, 0.0, 84.45573745266202, -16.39546225863475, 0.0, 0.0},
                {0.0, -2.8210572700220213, -16.39546225863475, 35.1238772454151, 0.0, 0.0},
                {2.8210572700220213, 0.0, 0.0, 0.0, 3.616337664050583, 0.6232077662449509},
                {16.39546225863475, 0.0, 0.0, 0.0, 0.6232077662449509, 31.507539581364625},
            }},
            {{
                {4521289678.947009, 0.0, 0.0, 0.0, 148760396.98544225, 266089109.4029744},
                {0.0, 239673891.33303756, -993432.1718331773, -10738401.632739767, 0.0, 0.0},
                {0.0, -993432.1718194059, 27272722.33611301, 3331136.312449322, 0.0, 0.0},
                {0.0, -10738401.632749403, 3331136.3124463274, 23330158.861795776, 0.0, 0.0},
                {148760396.9854658, 0.0, 0.0, 0.0, 243066553.44047254, 4841801.236697684},
                {266089109.40288928, 0.0, 0.0, 0.0, 4841801.236680448, 683571619.6771157},
            }},
        },
        BeamSection{
            0.9,
            {{
                {54.70473032506861, 0.0, 0.0, 0.0, 1.6418254028409995, 11.033185396820333},
                {0.0, 54.70473032506861, 0.0, -1.6418254028409995, 0.0, 0.0},
                {0.0, 0.0, 54.70473032506861, -11.033185396820333, 0.0, 0.0},
                {0.0, -1.6418254028409995, -11.033185396820333, 21.103193236866915, 0.0, 0.0},
                {1.6418254028409995, 0.0, 0.0, 0.0, 1.78333715736047, 0.36848755076490375},
                {11.033185396820333, 0.0, 0.0, 0.0, 0.36848755076490375, 19.31985607950643},
            }},
            {{
                {2675615584.13509, 0.0, 0.0, 0.0, 79832450.74825534, 184789087.68407992},
                {0.0, 144312493.8658793, -510321.80042168766, -6016948.89773326, 0.0, 0.0},
                {0.0, -510321.8004162957, 19745236.354008205, 1832045.8991628585, 0.0, 0.0},
                {0.0, -6016948.89772213, 1832045.8991661698, 13417874.462175902, 0.0, 0.0},
                {79832450.7482741, 0.0, 0.0, 0.0, 113253467.97005808, 4125365.2248314293},
                {184789087.6841325, 0.0, 0.0, 0.0, 4125365.2248271434, 420297066.7672982},
            }},
        },
        BeamSection{
            0.95,
            {{
                {34.44992885232661, 0.0, 0.0, 0.0, 0.916505016263078, 8.119492797706668},
                {0.0, 34.44992885232661, 0.0, -0.916505016263078, 0.0, 0.0},
                {0.0, 0.0, 34.44992885232661, -8.119492797706668, 0.0, 0.0},
                {0.0, -0.916505016263078, -8.119492797706668, 13.020205616048557, 0.0, 0.0},
                {0.916505016263078, 0.0, 0.0, 0.0, 0.7573836161406937, 0.24956166062622986},
                {8.119492797706668, 0.0, 0.0, 0.0, 0.24956166062622986, 12.262821999907853},
            }},
            {{
                {1203040036.154266, 0.0, 0.0, 0.0, 31896540.37930602, 137345812.75625196},
                {0.0, 86939021.67477001, -307950.61736255157, -3396279.8195240535, 0.0, 0.0},
                {0.0, -307950.61736175837, 9179700.675681142, 1267110.0781497026, 0.0, 0.0},
                {0.0, -3396279.819523402, 1267110.0781552165, 6805118.090718728, 0.0, 0.0},
                {31896540.37929474, 0.0, 0.0, 0.0, 36395660.02090348, 3561131.7253238275},
                {137345812.75628257, 0.0, 0.0, 0.0, 3561131.725327512, 231512293.70712498},
            }},
        },
        BeamSection{
            1.0,
            {{
                {5.394970691335722, 0.0, 0.0, 0.0, 0.036961089182658016, 0.1588859366089294},
                {0.0, 5.394970691335722, 0.0, -0.036961089182658016, 0.0, 0.0},
                {0.0, 0.0, 5.394970691335722, -0.1588859366089294, 0.0, 0.0},
                {0.0, -0.036961089182658016, -0.1588859366089294, 0.10091508735469075, 0.0, 0.0},
                {0.036961089182658016, 0.0, 0.0, 0.0, 0.007180810431802968, 0.0022768326051883147},
                {0.1588859366089294, 0.0, 0.0, 0.0, 0.0022768326051883147, 0.09373427692288798},
            }},
            {{
                {118283403.93224278, 0.0, 0.0, 0.0, 795943.4168930704, 1168848.1431805997},
                {0.0, 15879263.61054028, -232415.29456330693, -117375.6660814398, 0.0, 0.0},
                {0.0, -232415.29456365184, 931474.1817045894, 106353.52899654783, 0.0, 0.0},
                {0.0, -117375.66608144647, 106353.52899656125, 71452.33631373737, 0.0, 0.0},
                {795943.416893326, 0.0, 0.0, 0.0, 186239.91293931668, 26483.6917424594},
                {1168848.1431801969, 0.0, 0.0, 0.0, 26483.691742438405, 1384839.7699354952},
            }},
        },
    };

    // Build vector of nodes (straight along x axis, no rotation)
    // Calculate displacement, velocity, acceleration assuming a
    // 1 rad/s angular velocity around the z axis
    const size_t num_blades = 3;
    std::vector<BeamElement> blade_elems;
    std::vector<std::array<double, 7>> displacement;
    std::vector<std::array<double, 6>> velocity;
    std::vector<std::array<double, 6>> acceleration;
    std::vector<ConstraintInput> constraint_inputs;

    // Hub radius (meters)
    const double hub_rad(3.97);

    // Loop through blades
    for (size_t i = 0; i < num_blades; ++i) {
        // Define root rotation
        auto q_root = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(
            Vector(0, 0, 2.0 * M_PI * i / num_blades)
        );

        // Declare list of element nodes
        std::vector<BeamNode> nodes;

        // Loop through nodes
        for (size_t j = 0; j < node_loc.size(); ++j) {
            auto pos = q_root * (node_coords[j] + Vector(hub_rad, 0, 0));
            auto rot = q_root * node_rotation[j];
            nodes.push_back(BeamNode(node_loc[j], pos, rot));

            // Add node initial displacement, velocity, and acceleration
            displacement.push_back({0., 0., 0., 1., 0., 0., 0.});
            auto v = omega.CrossProduct(pos);
            velocity.push_back({
                v.GetX(),
                v.GetY(),
                v.GetZ(),
                omega.GetXComponent(),
                omega.GetYComponent(),
                omega.GetZComponent(),
            });
            acceleration.push_back({0., 0., 0., 0., 0., 0.});
        }

        // Add beam element
        blade_elems.push_back(BeamElement(nodes, material_sections, trapz_quadrature));

        // Set constraint nodes
        constraint_inputs.push_back(ConstraintInput(-1, i * node_loc.size()));
    }

    // Define beam initialization
    BeamsInput beams_input(blade_elems, gravity);

    // Initialize beams from element inputs
    auto beams = CreateBeams(beams_input);

    // Number of system nodes from number of beam nodes
    const size_t num_system_nodes(beams.num_nodes);

    // Create solver with initial node state
    Solver solver(
        is_dynamic_solve, max_iter, step_size, rho_inf, num_system_nodes, constraint_inputs,
        displacement, velocity, acceleration
    );

    // Initialize constraints
    InitializeConstraints(solver, beams);

    // Perform time steps and check for convergence within max_iter iterations
    for (size_t i = 0; i < num_steps; ++i) {
        // Calculate hub rotation for this time step
        auto q_hub = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(
            omega * step_size * (i + 1)
        );

        // Define hub translation/rotation displacement
        Array_7 u_hub(
            {0, 0, 0, q_hub.GetScalarComponent(), q_hub.GetXComponent(), q_hub.GetYComponent(),
             q_hub.GetZComponent()}
        );

        // Update constraint displacements
        for (size_t j = 0; j < solver.num_constraint_nodes; ++j) {
            solver.constraints.UpdateDisplacement(j, u_hub);
        }

        // Take step
        auto converged = Step(solver, beams);

        // Verify that step converged
        EXPECT_EQ(converged, true);
    }
}

}  // namespace openturbine::restruct_poc::tests
