#include "src/gebt_poc/state.h"

#include <KokkosBlas.hpp>

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

State::State()
    : generalized_coords_("generalized_coordinates", 1),
      velocity_("velocities", 1),
      acceleration_("accelerations", 1),
      algorithmic_acceleration_("algorithmic_accelerations", 1) {
}

State::State(
    Kokkos::View<double* [kNumberOfLieGroupComponents]> gen_coords,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> velocity,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> accln,
    Kokkos::View<double* [kNumberOfLieAlgebraComponents]> algo_accln
)
    : generalized_coords_("generalized_coordinates", gen_coords.extent(0)),
      velocity_("velocities", velocity.extent(0)),
      acceleration_("accelerations", accln.extent(0)),
      algorithmic_acceleration_("algorithmic_accelerations", algo_accln.extent(0)) {
    auto n_gen_coords = gen_coords.extent(0);
    auto n_velocities = velocity.extent(0);
    auto n_accelerations = accln.extent(0);
    auto n_algo_accelerations = algo_accln.extent(0);

    if (n_gen_coords != n_velocities || n_gen_coords != n_accelerations ||
        n_gen_coords != n_algo_accelerations) {
        throw std::invalid_argument(
            "The number rows (i.e. nodes) of generalized coordinates, velocities, accelerations, "
            "and algorithmic accelerations in the initial state must be the same"
        );
    }

    Kokkos::deep_copy(generalized_coords_, gen_coords);
    Kokkos::deep_copy(velocity_, velocity);
    Kokkos::deep_copy(acceleration_, accln);
    Kokkos::deep_copy(algorithmic_acceleration_, algo_accln);
}

MassMatrix::MassMatrix(
    double mass, Kokkos::View<double[3]> center_of_mass, Kokkos::View<double[3][3]> moment_of_inertia
)
    : mass_(mass),
      center_of_mass_("center_of_mass"),
      moment_of_inertia_("moment_of_inertia"),
      mass_matrix_("mass_matrix") {
    if (mass <= 0.) {
        throw std::invalid_argument("Mass must be positive");
    }

    Kokkos::deep_copy(center_of_mass_, center_of_mass);
    Kokkos::deep_copy(moment_of_inertia_, moment_of_inertia);

    // The mass matrix is defined as follows in the material frame:
    // [M]_6x6 = [
    //      [m[I]_3x3]          [m[~{eta}^T]]
    //      [m [~{eta}]]          [rho]_3x3
    // ]
    // where,
    // [I]_3x3 = identity matrix [1]_3x3
    // {eta} = a position vector for the center of mass and ~{eta} = cross product matrix of {eta}
    // [rho]_3x3 = moment of inertia matrix

    Kokkos::deep_copy(mass_matrix_, 0.);
    auto m = Kokkos::subview(mass_matrix_, Kokkos::make_pair(0, 3), Kokkos::make_pair(0, 3));
    auto m_eta_T = Kokkos::subview(mass_matrix_, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6));
    auto m_eta = Kokkos::subview(mass_matrix_, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
    auto rho = Kokkos::subview(mass_matrix_, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));

    // Top left quadrant
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            m(0, 0) = mass_;
            m(1, 1) = mass_;
            m(2, 2) = mass_;
        }
    );

    // Top right quadrant
    auto eta_cross_prod_matrix = gen_alpha_solver::create_cross_product_matrix(center_of_mass_);
    KokkosBlas::scal(m_eta, mass_, eta_cross_prod_matrix);

    // Bottom left quadrant
    KokkosBlas::gemm("T", "N", 1., eta_cross_prod_matrix, m, 0., m_eta_T);

    // Bottom right quadrant
    Kokkos::deep_copy(rho, moment_of_inertia_);
}

MassMatrix::MassMatrix(Kokkos::View<double[6][6]> mass_matrix)
    : mass_(0.),
      center_of_mass_("center_of_mass"),
      moment_of_inertia_("moment_of_inertia"),
      mass_matrix_("mass_matrix") {
    Kokkos::deep_copy(mass_matrix_, mass_matrix);

    // Calculate mass, center of mass, and moment of inertia from the mass matrix
    auto m = Kokkos::subview(mass_matrix_, Kokkos::make_pair(0, 1), Kokkos::make_pair(0, 1));
    this->mass_ = m(0, 0);

    auto m_eta = Kokkos::subview(mass_matrix_, Kokkos::make_pair(3, 6), Kokkos::make_pair(0, 3));
    auto temp = Kokkos::View<double[3][3]>("temp");
    KokkosBlas::scal(temp, 1. / this->mass_, m_eta);
    center_of_mass_ = gen_alpha_solver::convert_cross_product_matrix_to_vector(temp);

    auto rho = Kokkos::subview(mass_matrix_, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(moment_of_inertia_, rho);
}

GeneralizedForces::GeneralizedForces(
    const gen_alpha_solver::Vector& forces, const gen_alpha_solver::Vector& moments, size_t node
)
    : forces_(forces), moments_(moments), generalized_forces_("generalized_forces"), node_(node) {
    Kokkos::parallel_for(
        1,
        KOKKOS_LAMBDA(std::size_t) {
            generalized_forces_(0) = forces.GetXComponent();
            generalized_forces_(1) = forces.GetYComponent();
            generalized_forces_(2) = forces.GetZComponent();
            generalized_forces_(3) = moments.GetXComponent();
            generalized_forces_(4) = moments.GetYComponent();
            generalized_forces_(5) = moments.GetZComponent();
        }
    );
}

GeneralizedForces::GeneralizedForces(Kokkos::View<double[6]> generalized_forces, size_t node)
    : generalized_forces_("generalized_forces"), node_(node) {
    Kokkos::deep_copy(generalized_forces_, generalized_forces);

    auto forces = Kokkos::subview(generalized_forces_, Kokkos::make_pair(0, 3));
    auto moments = Kokkos::subview(generalized_forces_, Kokkos::make_pair(3, 6));
    this->forces_ = gen_alpha_solver::Vector(forces(0), forces(1), forces(2));
    this->moments_ = gen_alpha_solver::Vector(moments(0), moments(1), moments(2));
}

}  // namespace openturbine::gebt_poc
