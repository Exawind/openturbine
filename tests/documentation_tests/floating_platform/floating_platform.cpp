#include <array>
#include <cassert>

#include <Kokkos_Core.hpp>
#include <interfaces/cfd/interface.hpp>

int main() {
    Kokkos::initialize();
    {
        // Solution parameters
        constexpr auto time_step = 0.01;  // Time step size
        constexpr auto t_end = 1.;        // Final time of simulation
        constexpr auto rho_inf = 0.0;     // Time stepping damping factor
        constexpr auto max_iter = 5;      // Maximum number of nonlinear steps per time ste[
        const auto n_steps{
            static_cast<size_t>(ceil(t_end / time_step)) + 1
        };  // Number of time steps

        // Define gravity vector
        constexpr auto gravity = std::array{0., 0., -9.8124};  // m/s/s

        // Construct platform mass matrix as a 6x6 "array of arrays"
        constexpr auto platform_mass{1.419625E+7};                                     // kg
        constexpr auto platform_moi = std::array{1.2898E+10, 1.2851E+10, 1.4189E+10};  // kg*m*m
        constexpr auto platform_cm_position = std::array{0., 0., -7.53};               // m
        constexpr auto platform_mass_matrix = std::array{
            std::array{platform_mass, 0., 0., 0., 0., 0.},    // Row 1
            std::array{0., platform_mass, 0., 0., 0., 0.},    // Row 2
            std::array{0., 0., platform_mass, 0., 0., 0.},    // Row 3
            std::array{0., 0., 0., platform_moi[0], 0., 0.},  // Row 4
            std::array{0., 0., 0., 0., platform_moi[1], 0.},  // Row 5
            std::array{0., 0., 0., 0., 0., platform_moi[2]},  // Row 6
        };

        // Define mooring line stiffness and initial length
        constexpr auto mooring_line_stiffness{48.9e3};       // N
        constexpr auto mooring_line_initial_length{55.432};  // m

        // Create cfd interface
        auto interface = openturbine::cfd::Interface(openturbine::cfd::InterfaceInput{
            gravity,
            time_step,  // time step
            rho_inf,    // rho infinity (numerical damping)
            max_iter,   // max convergence iterations
            openturbine::cfd::TurbineInput{
                openturbine::cfd::FloatingPlatformInput{
                    true,  // enable
                    {
                        platform_cm_position[0],
                        platform_cm_position[1],
                        platform_cm_position[2],
                        1.,
                        0.,
                        0.,
                        0.,
                    },                         // position
                    {0., 0., 0., 0., 0., 0.},  // velocity
                    {0., 0., 0., 0., 0., 0.},  // acceleration
                    platform_mass_matrix,
                    {
                        {
                            mooring_line_stiffness,
                            mooring_line_initial_length,
                            {-40.87, 0.0, -14.},    // Fairlead node coordinates
                            {0., 0., 0.},           // Fairlead node velocity
                            {0., 0., 0.},           // Fairlead node acceleration
                            {-105.47, 0.0, -58.4},  // Anchor node coordinates
                            {0., 0., 0.},           // Anchor node velocity
                            {0., 0., 0.},           // Anchor node acceleration
                        },
                        {
                            mooring_line_stiffness,
                            mooring_line_initial_length,
                            {20.43, -35.39, -14.},   // Fairlead node coordinates
                            {0., 0., 0.},            // Fairlead node velocity
                            {0., 0., 0.},            // Fairlead node acceleration
                            {52.73, -91.34, -58.4},  // Anchor node coordinates
                            {0., 0., 0.},            // Anchor node velocity
                            {0., 0., 0.},            // Anchor node acceleration
                        },
                        {
                            mooring_line_stiffness,
                            mooring_line_initial_length,
                            {20.43, 35.39, -14.},   // Fairlead node coordinates
                            {0., 0., 0.},           // Fairlead node velocity
                            {0., 0., 0.},           // Fairlead node acceleration
                            {52.73, 91.34, -58.4},  // Anchor node coordinates
                            {0., 0., 0.},           // Anchor node velocity
                            {0., 0., 0.},           // Anchor node acceleration
                        },
                    },
                },
            },
        });

        // Save the initial state, then take first step
        interface.SaveState();
        auto converged = interface.Step();

        // Calculate buoyancy force as percentage of gravitational force plus spring forces times
        const auto spring_f = Kokkos::create_mirror(interface.elements.springs.f);
        Kokkos::deep_copy(spring_f, interface.elements.springs.f);
        const auto initial_spring_force = spring_f(0, 2) + spring_f(1, 2) + spring_f(2, 2);
        const auto platform_gravity_force = -gravity[2] * platform_mass;
        const auto buoyancy_force = initial_spring_force + platform_gravity_force;

        // Reset to initial state and apply
        interface.RestoreState();

        // Iterate through time steps
        for (size_t i = 0U; i < n_steps; ++i) {
            // Calculate current time
            const auto t = static_cast<double>(i) * time_step;

            // Apply load in y direction
            interface.turbine.floating_platform.node.loads[1] = 1e6 * sin(2. * M_PI / 20. * t);

            // Apply time varying buoyancy force
            interface.turbine.floating_platform.node.loads[2] =
                buoyancy_force + 0.5 * initial_spring_force * sin(2. * M_PI / 20. * t);

            // Apply time varying moments to platform node
            interface.turbine.floating_platform.node.loads[3] =
                5.0e5 * sin(2. * M_PI / 15. * t);  // rx
            interface.turbine.floating_platform.node.loads[4] =
                1.0e6 * sin(2. * M_PI / 30. * t);  // ry
            interface.turbine.floating_platform.node.loads[5] =
                2.0e7 * sin(2. * M_PI / 60. * t);  // rz

            // Take a single time step
            converged = interface.Step();
            assert(converged);
        }
    }
    Kokkos::finalize();
    return 0;
}
