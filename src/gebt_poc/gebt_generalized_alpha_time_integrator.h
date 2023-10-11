#pragma once

#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_fill.hpp>
#include <KokkosBlas3_gemm.hpp>
 
#include "src/gen_alpha_poc/solver.h"
 
#include "src/gebt_poc/mesh.h"
#include "src/gebt_poc/field_data.h"

namespace openturbine::gebt_poc {

class LinearizationParameters {
public:
virtual void ComputeResidualVector(Kokkos::View<double*> residuals, Mesh& mesh, FieldData& field_data, Kokkos::View<double*> lagrange_mults) = 0;

virtual void ComputeIterationMatrix(Kokkos::View<double**> iteration_matrix, double h, double betaPrime, double gammaPrime, Mesh& mesh, FieldData& field_data, Kokkos::View<double*> lagrange_mults) = 0;
};

class UnityLinearizationParameters : public LinearizationParameters{
public:
void ComputeResidualVector(Kokkos::View<double*> residuals, Mesh& mesh, FieldData& field_data, Kokkos::View<double*> lagrange_mults) {
  KokkosBlas::fill(residuals, 1.);
}

void ComputeIterationMatrix(Kokkos::View<double**> iteration_matrix, double h, double betaPrime, double gammaPrime, Mesh& mesh, FieldData& field_data, Kokkos::View<double*> lagrange_mults) {
  Kokkos::parallel_for(iteration_matrix.extent(0), KOKKOS_LAMBDA(std::size_t i) {
    for(std::size_t j = 0; j < iteration_matrix.extent(1); ++j) {
      iteration_matrix(i, j) = (i == j);
    }
  });
}

};

KOKKOS_FUNCTION
void exponential_mapping_with_scale(Kokkos::View<double*> quaternion, Kokkos::View<double*> vector, double scale) {
  auto a = openturbine::gen_alpha_solver::Vector(vector(0) * scale, vector(1) * scale, vector(2) * scale);
  auto b = openturbine::gen_alpha_solver::quaternion_from_rotation_vector(a);
  quaternion(0) = b.GetScalarComponent();
  quaternion(1) = b.GetXComponent();
  quaternion(2) = b.GetYComponent();
  quaternion(3) = b.GetZComponent();
}

KOKKOS_FUNCTION
void quaternion_mult(Kokkos::View<double*> out, Kokkos::View<double*> q1, Kokkos::View<double*> q2) {
  auto a = openturbine::gen_alpha_solver::Quaternion(q1(0), q1(1), q1(2), q1(3));
  auto b = openturbine::gen_alpha_solver::Quaternion(q2(0), q2(1), q2(2), q2(3));
  auto c = a * b;
  out(0) = c.GetScalarComponent();
  out(1) = c.GetXComponent();
  out(2) = c.GetYComponent();
  out(3) = c.GetZComponent();
}

class GeneralizedAlphaStepper {
public:
  bool Step(Mesh& mesh, FieldData& field_data, std::size_t lagrange_multipliers, double time_step_size, std::size_t max_iterations) {
    constexpr auto lie_algebra_size = 6;
    constexpr auto lie_group_size = 7;
    auto alphaF = alphaF_;
    auto alphaM = alphaM_;
    auto beta = beta_;
    auto gamma = gamma_;
    auto h = time_step_size;

    auto num_nodes = mesh.GetNumberOfNodes();
    auto number_of_constraints = lagrange_multipliers;
    auto dof_size = lie_algebra_size * num_nodes;
    auto residual_size = dof_size + number_of_constraints;

    auto lagrange_mults = Kokkos::View<double*>("lagrange_mults", lagrange_multipliers);
    auto residuals = Kokkos::View<double*>("residual", residual_size);
    auto iteration_matrix = Kokkos::View<double**>("iteration matrix", residual_size, residual_size);
    auto left_pre = Kokkos::View<double**>("left preconditioner", residual_size, residual_size);
    auto right_pre = Kokkos::View<double**>("left preconditioner", residual_size, residual_size);
    
    auto dof_residuals = Kokkos::subview(residuals, Kokkos::pair<int,int>(0, dof_size));
    auto lagrange_residuals = Kokkos::subview(residuals, Kokkos::pair<int,int>(dof_size, residual_size));

    Kokkos::parallel_for(mesh.GetNumberOfNodes(), KOKKOS_LAMBDA(std::size_t node) {
      auto acceleration = field_data.GetNodalData<Field::Acceleration>(node);
      auto algo_acceleration_next = field_data.GetNodalData<Field::AlgorithmicAccelerationNext>(node);
      auto delta_coordinates = field_data.GetNodalData<Field::DeltaCoordinates>(node);
      auto velocity = field_data.GetNodalData<Field::Velocity>(node);
      auto algo_acceleration = field_data.GetNodalData<Field::AlgorithmicAcceleration>(node);

      for(int i = 0; i < lie_algebra_size; ++i) {
        algo_acceleration_next(i) = (alphaF * acceleration(i) - alphaM * algo_acceleration(i)) / (1. - alphaM);
        delta_coordinates(i) = velocity(i) + h * (.5 - beta) * algo_acceleration(i) + h * beta * algo_acceleration_next(i);
        velocity(i) += h * (1. - gamma) * algo_acceleration(i) + h * gamma * algo_acceleration_next(i);
        algo_acceleration(i) = algo_acceleration_next(i);
        acceleration(i) = 0.;
      }
    });
    Kokkos::deep_copy(lagrange_mults, 0.);

    auto betaPrime = (1. - alphaM) / (h * h * beta *(1 - alphaF));
    auto gammaPrime = gamma / (h * beta);
    auto scalar_pre = (is_preconditioned_) ? beta * h * h : 1.;

    Kokkos::deep_copy(left_pre, 0.);
    Kokkos::deep_copy(right_pre, 0.);
    Kokkos::parallel_for(residual_size, KOKKOS_LAMBDA(int i) {
      left_pre(i, i) = (i < dof_size) ? scalar_pre : 1.;
      right_pre(i, i) = (i < dof_size) ? 1. : 1. / scalar_pre;
    });

    std::size_t nonlinear_iteration;
    bool is_converged = false;
    for(nonlinear_iteration = 0; nonlinear_iteration < max_iterations; ++nonlinear_iteration) {
      Kokkos::parallel_for(mesh.GetNumberOfNodes(), KOKKOS_LAMBDA(std::size_t node) {
        auto coordinates_next = field_data.GetNodalData<Field::CoordinatesNext>(node);
        auto coordinates = field_data.GetNodalData<Field::Coordinates>(node);
        auto delta_coordinates = field_data.GetNodalData<Field::DeltaCoordinates>(node);
        for(int i = 0; i < 3; ++i) {
          coordinates_next(i) = coordinates(i) + delta_coordinates(i) * h;
        } 
  
        auto orientation_vector = Kokkos::subview(delta_coordinates, Kokkos::pair<int,int>(3,6));
        auto updated_orientation = Kokkos::View<double[4]>("updated_orientation");
        exponential_mapping_with_scale(updated_orientation, orientation_vector, h); //must use orientation_vector * h
        auto current_orientation = Kokkos::subview(coordinates, Kokkos::pair<int,int>(3,7));
        auto next_orientation = Kokkos::subview(coordinates_next, Kokkos::pair<int,int>(3,7));
        quaternion_mult(next_orientation, current_orientation, updated_orientation);
      });

      assembler_->ComputeResidualVector(residuals, mesh, field_data, lagrange_mults);
      
      if((is_converged = CheckConvergence(residuals))) {
        return is_converged;
      }

      assembler_->ComputeIterationMatrix(iteration_matrix, h, betaPrime, gammaPrime, mesh, field_data, lagrange_mults);

      KokkosBlas::gemm("N", "N", 1., iteration_matrix, right_pre, 0., iteration_matrix);
      KokkosBlas::gemm("N", "N", 1., left_pre, iteration_matrix, 0., iteration_matrix);
      KokkosBlas::scal(dof_residuals, scalar_pre, dof_residuals);
 
      openturbine::gen_alpha_solver::solve_linear_system(iteration_matrix, residuals);

      KokkosBlas::scal(residuals, -1., residuals);

      KokkosBlas::axpby(-1. / scalar_pre, lagrange_residuals, 1., lagrange_mults);

      Kokkos::parallel_for(mesh.GetNumberOfNodes(), KOKKOS_LAMBDA(std::size_t node) {
        auto delta_coordinates = field_data.GetNodalData<Field::DeltaCoordinates>(node);
        auto velocity = field_data.GetNodalData<Field::Velocity>(node);
        auto acceleration = field_data.GetNodalData<Field::Acceleration>(node);

        int start_index = node * lie_algebra_size;
        int end_index = start_index + lie_algebra_size;
        auto update = Kokkos::subview(dof_residuals, Kokkos::pair<int,int>(start_index, end_index));
        for(int i = 0; i < lie_algebra_size; ++i) {
          delta_coordinates(i) += update(i) / h;
          velocity(i) += gammaPrime * update(i);
          acceleration(i) += betaPrime * update(i);
        }
      });
    }

    Kokkos::parallel_for(mesh.GetNumberOfNodes(), KOKKOS_LAMBDA(std::size_t node) {
      auto algo_acceleration_next = field_data.GetNodalData<Field::AlgorithmicAccelerationNext>(node);
      auto acceleration = field_data.GetNodalData<Field::Acceleration>(node);
      for(std::size_t i = 0; i < lie_algebra_size; ++i) {
        algo_acceleration_next(i) += (1. - alphaF) / (1. - alphaM) * acceleration(i);
      }

      auto algo_acceleration = field_data.GetNodalData<Field::AlgorithmicAcceleration>(node);
      for(std::size_t i = 0; i < lie_algebra_size; ++i) {
        algo_acceleration(i) = algo_acceleration_next(i);
      }

      auto coordinates = field_data.GetNodalData<Field::Coordinates>(node);
      auto coordinates_next = field_data.GetNodalData<Field::CoordinatesNext>(node);
      for(std::size_t i = 0; i < lie_group_size; ++i) {
        coordinates(i) = coordinates_next(i);
      }
    });
    return is_converged;
  }


  friend GeneralizedAlphaStepper CreateBasicStepper();

protected:
  GeneralizedAlphaStepper() = default;

  void SetParameters(double alpha_f, double alpha_m, double beta, double gamma) {
   alphaF_ = alpha_f;
   alphaM_ = alpha_m;
   beta_ = beta;
   gamma_ = gamma;
  }

  void SetPreconditioner(bool is_preconditioned) {
    is_preconditioned_ = is_preconditioned;
  }

  void SetSystemAssembler(std::shared_ptr<LinearizationParameters> assembler) {
    assembler_ = assembler;
  }

  bool CheckConvergence(Kokkos::View<double*> residuals) {
    return KokkosBlas::nrm2(residuals) < 1.e-10;
  }

  std::shared_ptr<LinearizationParameters> assembler_;

  double alphaF_;
  double alphaM_;
  double beta_;
  double gamma_;
  bool is_preconditioned_;
};

GeneralizedAlphaStepper CreateBasicStepper() {
  GeneralizedAlphaStepper stepper;
  stepper.SetParameters(0., 0., .5, 1.);
  stepper.SetPreconditioner(false);

  stepper.SetSystemAssembler(std::make_shared<UnityLinearizationParameters>());

  return stepper;
}

}
