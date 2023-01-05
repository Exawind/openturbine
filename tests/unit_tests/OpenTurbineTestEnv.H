#ifndef OPENTURBINETESTENV_H
#define OPENTURBINETESTENV_H

#include "gtest/gtest.h"

namespace openturbine_tests {

/** Global setup/teardown test enviroment configuration for OpenTurbine unit
 * tests
 *
 *  This class is registered with GoogleTest infrastructure to perform global
 *  setup/teardown tasks.
 *
 *  This was adapted from
 * https://github.com/Exawind/amr-wind/blob/main/unit_tests/aw_test_utils/AmrexTestEnv.H
 *  accessed on December 11, 2020.
 *
 */
class OTurbTestEnv : public ::testing::Environment
{
public:
    OTurbTestEnv(int& argc, char**& argv) : m_argc(argc), m_argv(argv) {}

    ~OTurbTestEnv() override = default;

    void SetUp() override
    {
        // Do any setup here
    }

    void TearDown() override
    {
        // Do any clean up here
    }

protected:
    int& m_argc;
    char**& m_argv;
};

} // namespace openturbine_tests

#endif /* OPENTURBINETESTENV_H */