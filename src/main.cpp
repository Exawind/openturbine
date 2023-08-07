#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include <Kokkos_Core.hpp>

#include "src/OpenTurbineVersion.H"
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

    Kokkos::initialize(argc, argv);

    std::cout << "Hello from Open Turbine! (Note to Faisal: What should the program do here?)" << std::endl;

    Kokkos::finalize();

    return 0;
}
