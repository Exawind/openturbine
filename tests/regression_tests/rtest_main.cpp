/** \file utest_main.cpp
 *  Entry point for unit tests
 */

#include <Kokkos_Core.hpp>

#include "gtest/gtest.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    Kokkos::initialize(argc, argv);
    const int test_status = RUN_ALL_TESTS();
    Kokkos::finalize();

    return test_status;
}
