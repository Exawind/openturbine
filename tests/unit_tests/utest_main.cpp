/** \file utest_main.cpp
 *  Entry point for unit tests
 */

#include "OpenTurbineTestEnv.H"
#include "gtest/gtest.h"
#include <Kokkos_Core.hpp>

//! Global instance of the environment (for access in tests)
openturbine_tests::OTurbTestEnv* utest_env{nullptr};

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    utest_env = new openturbine_tests::OTurbTestEnv(argc, argv);
    ::testing::AddGlobalTestEnvironment(utest_env);

    Kokkos::initialize(argc, argv);
    int test_status = RUN_ALL_TESTS();
    Kokkos::finalize();

    return test_status;
}
