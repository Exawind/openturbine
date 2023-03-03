#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <Kokkos_Core.hpp>

#include "src/OpenTurbineVersion.H"
#include "src/heat/heat_solve.H"
#include "src/io/console_io.H"
#include "src/utilities/debug_utils.H"
#include "src/utilities/log.h"

int main(int argc, char* argv[]) {
    using namespace openturbine;

    if (argc > 2) {
        // Print usage and exit with error code if no input file was provided.
        io::print_usage(std::cout);
        io::print_error("No input file provided. Exiting.");
        return 1;
    }

    // cppcheck-suppress knownConditionTrueFalse
    // Look for "-h" or "--help" flag and print usage
    for (auto i = 1; i < argc; i++) {
        const std::string param(argv[i]);
        if ((param == "--help") || (param == "-h")) {
            io::print_banner(std::cout);
            io::print_usage(std::cout);
            return 0;
        }
    }

    // TODO Name the logging file based on the provided input file name
    std::string log_file = "log.txt";
    if (std::filesystem::exists(log_file)) {
        std::cout << "Overwriting the previously existing log file" << std::endl;
    }
    std::ofstream{log_file};

#ifdef DEBUG
    auto log = util::Log::Get(log_file, util::SeverityLevel::kDebug);
#elif defined RELEASE
    auto log = util::Log::Get(log_file, util::SeverityLevel::kInfo);
#else
    auto log = util::Log::Get(log_file, util::SeverityLevel::kNone);
#endif

    log->Info("openturbine " + version::oturb_version + "\n");

    int axis_size{5};                 // Size of the 1D grid
    double side_length{1.0};          // Length of the 1D domain
    double k{1.011};                  // Thermal diffusivity of the material in units of cm^2/s
    double ic_x0{100.0};              // Initial temperature at the beginning of the domain
    double ic_x1{100.0};              // Initial temperature at the end of the domain
    int n_max{500};                   // Max iterations
    double residual_tolerance{1e-5};  // Iterative residual tolerance

    auto U = static_cast<double*>(std::malloc(axis_size * sizeof(double)));

    Kokkos::initialize(argc, argv);
    heat_solve::heat_conduction_solver(axis_size, side_length, n_max, k, ic_x0, ic_x1,
                                       residual_tolerance);
    Kokkos::finalize();

    util::print_array(U, axis_size);

    std::free(U);

    return 0;
}
