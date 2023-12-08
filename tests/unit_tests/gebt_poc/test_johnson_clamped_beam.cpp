#include <gtest/gtest.h>

#include "src/gebt_poc/gen_alpha_2D.h"
#include "src/gebt_poc/solver.h"
#include "src/gebt_poc/static_beam_element.h"
#include "tests/unit_tests/gen_alpha_poc/test_utilities.h"

namespace openturbine::gebt_poc::tests {

struct PopulatePositionVectors {
    Kokkos::View<double[35]> position_vectors;
    KOKKOS_FUNCTION
    void operator()(std::size_t) const {
        // node 1
        position_vectors(0) = 1.;
        position_vectors(1) = 0.;
        position_vectors(2) = 0.;
        position_vectors(3) = 1.;
        position_vectors(4) = 0.;
        position_vectors(5) = 0.;
        position_vectors(6) = 0.;
        // node 2
        position_vectors(7) = 2.7267316464601146;
        position_vectors(8) = 0.;
        position_vectors(9) = 0.;
        position_vectors(10) = 1.;
        position_vectors(11) = 0.;
        position_vectors(12) = 0.;
        position_vectors(13) = 0.;
        // node 3
        position_vectors(14) = 6.;
        position_vectors(15) = 0.;
        position_vectors(16) = 0.;
        position_vectors(17) = 1.;
        position_vectors(18) = 0.;
        position_vectors(19) = 0.;
        position_vectors(20) = 0.;
        // node 4
        position_vectors(21) = 9.273268353539885;
        position_vectors(22) = 0.;
        position_vectors(23) = 0.;
        position_vectors(24) = 1.;
        position_vectors(25) = 0.;
        position_vectors(26) = 0.;
        position_vectors(27) = 0.;
        // node 5
        position_vectors(28) = 11.;
        position_vectors(29) = 0.;
        position_vectors(30) = 0.;
        position_vectors(31) = 1.;
        position_vectors(32) = 0.;
        position_vectors(33) = 0.;
        position_vectors(34) = 0.;
    }
};

}  // namespace openturbine::gebt_poc::tests
