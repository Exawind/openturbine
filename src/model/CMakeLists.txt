target_sources(openturbine_library PRIVATE)

install(FILES
        copy_nodes_to_state.hpp
        mesh_connectivity.hpp
        model.hpp
        node.hpp
        outputs.hpp
        DESTINATION include/OpenTurbine/model/
)
