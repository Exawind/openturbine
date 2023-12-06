#include "src/gen_alpha_poc/state.h"

namespace openturbine::gen_alpha_solver {

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

MassMatrix::MassMatrix(double mass, Vector J) : mass_(mass), principal_moment_of_inertia_(J) {
    if (mass <= 0.) {
        throw std::invalid_argument("Mass must be positive");
    }
    if (J.GetXComponent() <= 0. || J.GetYComponent() <= 0. || J.GetZComponent() <= 0.) {
        throw std::invalid_argument("Moment of inertia must be positive");
    }

    auto mass_matrix = std::vector<std::vector<double>>{
        {mass, 0., 0., 0., 0., 0.},               // row 1
        {0., mass, 0., 0., 0., 0.},               // row 2
        {0., 0., mass, 0., 0., 0.},               // row 3
        {0., 0., 0., J.GetXComponent(), 0., 0.},  // row 4
        {0., 0., 0., 0., J.GetYComponent(), 0.},  // row 5
        {0., 0., 0., 0., 0., J.GetZComponent()}   // row 6
    };
    this->mass_matrix_ = create_matrix(mass_matrix);
}

MassMatrix::MassMatrix(double mass, double moment_of_inertia)
    : MassMatrix(mass, {moment_of_inertia, moment_of_inertia, moment_of_inertia}) {
}

MassMatrix::MassMatrix(Kokkos::View<double**> mass_matrix)
    : mass_matrix_("mass_matrix", mass_matrix.extent(0), mass_matrix.extent(1)) {
    if (mass_matrix_.extent(0) != 6 || mass_matrix_.extent(1) != 6) {
        throw std::invalid_argument("Mass matrix must be 6 x 6");
    }
    Kokkos::deep_copy(mass_matrix_, mass_matrix);
    auto mass_matrix_host = Kokkos::create_mirror(mass_matrix_);
    Kokkos::deep_copy(mass_matrix_host, mass_matrix_);
    this->mass_ = mass_matrix_host(0, 0);
    this->principal_moment_of_inertia_ =
        Vector(mass_matrix_host(3, 3), mass_matrix_host(4, 4), mass_matrix_host(5, 5));
}

Kokkos::View<double**> MassMatrix::GetMomentOfInertiaMatrix() const {
    Kokkos::View<double**> moment_of_inertia_matrix("Moment of inertia matrix", 3, 3);

    constexpr int numComponents = 3;
    double J[numComponents] = {
        this->principal_moment_of_inertia_.GetXComponent(),
        this->principal_moment_of_inertia_.GetYComponent(),
        this->principal_moment_of_inertia_.GetZComponent()};

    Kokkos::parallel_for(
        "Moment of inertia matrix", numComponents,
        KOKKOS_LAMBDA(const int i) { moment_of_inertia_matrix(i, i) = J[i]; }
    );

    return moment_of_inertia_matrix;
}

GeneralizedForces::GeneralizedForces(const Vector& forces, const Vector& moments)
    : forces_(forces), moments_(moments) {
    this->generalized_forces_ = create_vector({
        forces.GetXComponent(),   // row 1
        forces.GetYComponent(),   // row 2
        forces.GetZComponent(),   // row 3
        moments.GetXComponent(),  // row 4
        moments.GetYComponent(),  // row 5
        moments.GetZComponent()   // row 6
    });
}

GeneralizedForces::GeneralizedForces(Kokkos::View<double*> generalized_forces)
    : generalized_forces_("generalized_forces_vector", generalized_forces.size()) {
    if (generalized_forces_.extent(0) != 6) {
        throw std::invalid_argument("Generalized forces must be 6 x 1");
    }
    Kokkos::deep_copy(generalized_forces_, generalized_forces);
    auto generalized_forces_host = Kokkos::create_mirror(generalized_forces_);
    Kokkos::deep_copy(generalized_forces_host, generalized_forces_);
    this->forces_ =
        Vector(generalized_forces_host(0), generalized_forces_host(1), generalized_forces_host(2));
    this->moments_ =
        Vector(generalized_forces_host(3), generalized_forces_host(4), generalized_forces_host(5));
}

}  // namespace openturbine::gen_alpha_solver
