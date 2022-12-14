#include <chrono>
#include <ctime>
#include <vector>
#include "src/utilities/console_io.H"
#include "src/OpenTurbineVersion.H"

namespace openturbine::io {

namespace {
const std::string dbl_line = std::string(78, '=') + "\n";
const std::string dash_line = "\n" + std::string(78, '-') + "\n";
} // namespace

void print_usage(std::ostream& out)
{

    out << R"doc(Usage:
    openturbine <input_file> [param=value] [param=value] ...

Required:
    input_file   : Input file with simulation settings

Optional:
    param=value  : Overrides for parameters during runtime
)doc" << std::endl;
}

void print_error(const std::string& msg)
{
    std::cout << "ERROR: " << msg << std::endl;
}

void print_banner(std::ostream& out)
{
    auto exec_time = std::chrono::system_clock::now();
    auto exect = std::chrono::system_clock::to_time_t(exec_time);
    const std::string dirty_tag = (version::oturb_dirty_repo == "DIRTY")
        ? ("-" + version::oturb_dirty_repo)
        : "";
    const std::string awind_version = version::oturb_version + dirty_tag;
    const std::string awind_git_sha = version::oturb_git_sha + dirty_tag;

    // clang-format off
    out << dbl_line
        << "                OpenTurbine (https://github.com/exawind/openturbine)"
        << std::endl << std::endl
        << "  OpenTurbine version :: " << awind_version << std::endl
        << "  OpenTurbine Git SHA :: " << awind_git_sha << std::endl
        << "  Exec. time       :: " << std::ctime(&exect)
        // << "  Build time       :: " << amrex::buildInfoGetBuildDate() << std::endl
        // << "  C++ compiler     :: " << amrex::buildInfoGetComp()
        // << " " << amrex::buildInfoGetCompVersion() << std::endl << std::endl
//         << "  GPU              :: "
// #ifdef AMREX_USE_GPU
//         << "ON    "
// #if defined(AMREX_USE_CUDA)
//         << "(Backend: CUDA)"
// #elif defined(AMREX_USE_HIP)
//         << "(Backend: HIP)"
// #elif defined(AMREX_USE_DPCPP)
//         << "(Backend: SYCL)"
// #endif
//         << std::endl
// #else
//         << "OFF" << std::endl
// #endif
//         << "  OpenMP           :: "
// #ifdef AMREX_USE_OMP
//         << "ON    (Num. threads = " << omp_get_max_threads() << ")" << std::endl
// #else
//         << "OFF" << std::endl
// #endif
        << std::endl;

    print_tpls(out);

    out << "           This software is released under the MIT License.                "
        << std::endl
        << " See https://github.com/Exawind/openturbine/blob/main/LICENSE for details. "
        << dash_line << std::endl;
    // clang-format on
}

void print_tpls(std::ostream& out)
{
    // TODO: Populate this with third party libraries

    std::vector<std::string> tpls;

    if (!tpls.empty()) {
        out << "  Enabled third-party libraries: ";
        for (const auto& val : tpls) {
            out << "\n    " << val;
        }
        out << std::endl << std::endl;
    } else {
        out << "  No additional third-party libraries enabled" << std::endl
            << std::endl;
    }
}

} // namespace openturbine::io