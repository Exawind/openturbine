#include <Kokkos_Core.hpp>

namespace openturbine::tests {
template<typename ValueType, typename DataType>
typename Kokkos::View<ValueType>::const_type CreateView(const std::string& name, const DataType& data) {
    const auto view = Kokkos::View<ValueType>(name);
    const auto host = typename Kokkos::View<ValueType, Kokkos::HostSpace>::const_type(data.data());
    const auto mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(mirror, host);
    Kokkos::deep_copy(view, mirror);
    return view;
}
}
