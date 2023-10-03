#pragma once

#include <Kokkos_Core.hpp>

#include "mesh.h"

namespace openturbine::gebt_poc {

enum class Field {
    Coordinates,
    Velocity,
    Acceleration,
    AlgorithmicAcceleration,
    CoordinatesNext,
    AlgorithmicAccelerationNext,
    DeltaCoordinates,
    Weight,
    StiffnessMatrix
};

class FieldData {
public:
    FieldData(const Mesh& mesh, int number_of_quadrature_points) {
        auto number_of_nodes = mesh.GetNumberOfNodes();
        auto number_of_elements = mesh.GetNumberOfElements();

        coordinates_ = Kokkos::View<double**>("coordinates", number_of_nodes, 3);
        velocity_ = Kokkos::View<double**>("velocity", number_of_nodes, 3);
        acceleration_ = Kokkos::View<double**>("acceleration", number_of_nodes, 3);
        algorithmic_acceleration_ =
            Kokkos::View<double**>("algorithmic_acceleration", number_of_nodes, 3);

        coordinates_next_ = Kokkos::View<double**>("coordinates_next", number_of_nodes, 3);
        algorithmic_acceleration_next_ =
            Kokkos::View<double**>("algorithmic_acceleration_next", number_of_nodes, 3);
        delta_coordinates_ = Kokkos::View<double**>("delta_coordinates", number_of_nodes, 3);

        weight_ = Kokkos::View<double**>("weight", number_of_elements, number_of_quadrature_points);
        stiffness_matrix_ = Kokkos::View<double****>(
            "stiffness", number_of_elements, number_of_quadrature_points, 6, 6
        );
    }

    template <Field field>
    KOKKOS_FUNCTION Kokkos::View<double*> GetNodalData(int node) const {
        if constexpr (field == Field::Coordinates) {
            return Kokkos::subview(coordinates_, node, Kokkos::ALL);
        } else if constexpr (field == Field::Velocity) {
            return Kokkos::subview(velocity_, node, Kokkos::ALL);
        } else if constexpr (field == Field::Acceleration) {
            return Kokkos::subview(acceleration_, node, Kokkos::ALL);
        } else if constexpr (field == Field::AlgorithmicAcceleration) {
            return Kokkos::subview(algorithmic_acceleration_, node, Kokkos::ALL);
        } else if constexpr (field == Field::CoordinatesNext) {
            return Kokkos::subview(coordinates_next_, node, Kokkos::ALL);
        } else if constexpr (field == Field::AlgorithmicAccelerationNext) {
            return Kokkos::subview(algorithmic_acceleration_next_, node, Kokkos::ALL);
        } else if constexpr (field == Field::DeltaCoordinates) {
            return Kokkos::subview(delta_coordinates_, node, Kokkos::ALL);
        } else
            Kokkos::abort("Provided Field is not a Nodal Field");
    }

    template <Field field>
    KOKKOS_FUNCTION Kokkos::View<const double*> ReadNodalData(int node) const {
        return GetNodalData<field>(node);
    }

    template <Field field>
    KOKKOS_FUNCTION auto GetElementData(int element) const {
        if constexpr (field == Field::Weight) {
            return Kokkos::subview(weight_, element, Kokkos::ALL);
        }
        if constexpr (field == Field::StiffnessMatrix) {
            return Kokkos::subview(
                stiffness_matrix_, element, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL
            );
        } else {
            Kokkos::abort("Provided Field is not an Element Field");
        }
    }

    template <Field field>
    KOKKOS_FUNCTION auto ReadElementData(int element) const {
        return GetElementData<field>(element);
    }

protected:
    Kokkos::View<double**> coordinates_;
    Kokkos::View<double**> velocity_;
    Kokkos::View<double**> acceleration_;
    Kokkos::View<double**> algorithmic_acceleration_;

    Kokkos::View<double**> coordinates_next_;
    Kokkos::View<double**> algorithmic_acceleration_next_;
    Kokkos::View<double**> delta_coordinates_;

    Kokkos::View<double**> weight_;
    Kokkos::View<double****> stiffness_matrix_;
};

}  // namespace openturbine::gebt_poc
