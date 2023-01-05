/** \file test_config.cpp
 *
 *  Tests various configurations for GPU builds
 */

#include <iostream>
#include "gtest/gtest.h"
#include "src/OpenTurbineVersion.H"

namespace oturb_tests {

TEST(Configuration, Build)
{
    const std::string dirty_tag =
        (openturbine::version::oturb_dirty_repo == "DIRTY")
            ? ("-" + openturbine::version::oturb_dirty_repo)
            : "";
    const std::string ot_git_sha =
        openturbine::version::oturb_git_sha + dirty_tag;
    std::cout << "OpenTurbine SHA = " << ot_git_sha << std::endl;
}

// TEST(Configuration, GPU)
// {

// #ifdef AMREX_USE_GPU
// #ifdef AMREX_USE_CUDA
//     amrex::Print() << "GPU backend: CUDA" << std::endl;
// #if defined(CUDA_VERSION)
//     amrex::Print() << "CUDA_VERSION: " << CUDA_VERSION << " "
//                    << CUDA_VERSION / 1000 << "." << (CUDA_VERSION % 1000) /
//                    10
//                    << std::endl;
// #endif
// #elif defined(AMREX_USE_HIP)
//     amrex::Print() << "GPU backend: HIP" << std::endl;
// #elif defined(AMREX_USE_DPCPP)
//     amrex::Print() << "GPU backend: SYCL" << std::endl;
// #endif

//     using Dev = amrex::Gpu::Device;
//     const int myrank = amrex::ParallelDescriptor::MyProc();
//     std::stringstream ss;
//     // clang-format off
//     ss << "[" << myrank << "] " << Dev::deviceId()
//         << ": " << Dev::deviceName() << "\n"
//         << "    Warp size          : " << Dev::warp_size << "\n"
//         << "    Global memory      : "
//         << (static_cast<double>(Dev::totalGlobalMem()) / (1 << 30)) << "GB\n"
//         << "    Shared mem/ block  : "
//         << (Dev::sharedMemPerBlock() / (1 << 10)) << "KB\n"
//         << "    Max. threads/block : " << Dev::maxThreadsPerBlock()
//         << " (" << Dev::maxThreadsPerBlock(0) << ", "
//         << Dev::maxThreadsPerBlock(1) << ", " << Dev::maxThreadsPerBlock(2)
//         << ")\n"
//         << "    Max. blocks/grid   : (" << Dev::maxBlocksPerGrid(0)
//         << ", " << Dev::maxBlocksPerGrid(1) << ", " <<
//         Dev::maxBlocksPerGrid(2) << ")\n"
//         << std::endl;
//     // clang-format on
//     amrex::OutStream() << ss.str();
// #else
//     amrex::Print() << "AMR-Wind not built with GPU support" << std::endl;
//     GTEST_SKIP();
// #endif
// }

TEST(Configuration, TPLs) {}

} // namespace oturb_tests
