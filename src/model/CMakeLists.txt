target_sources(openturbine_library PRIVATE)

install(FILES
        copy_nodes_to_state.hpp
        model.hpp
        node.hpp
        DESTINATION include/OpenTurbine/model/
)
