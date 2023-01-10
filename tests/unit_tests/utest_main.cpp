/** \file utest_main.cpp
 *  Entry point for unit tests
 */

#include "gtest/gtest.h"
#include "OpenTurbineTestEnv.H"

//! Global instance of the environment (for access in tests)
openturbine_tests::OTurbTestEnv* utest_env = nullptr;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    utest_env = new openturbine_tests::OTurbTestEnv(argc, argv);
    ::testing::AddGlobalTestEnvironment(utest_env);

    return RUN_ALL_TESTS();
}
