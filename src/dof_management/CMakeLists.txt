target_sources(openturbine_library 
  PRIVATE
)

install(FILES
        assemble_node_freedom_allocation_table.hpp
        compute_node_freedom_map_table.hpp
        create_constraint_freedom_table.hpp
        create_element_freedom_table.hpp
        freedom_signature.hpp
        DESTINATION include/OpenTurbine/dof_management
)
