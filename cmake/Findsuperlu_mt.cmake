set(SUPERLU_MT_FOUND FALSE)

set(SUPERLU_MT_INCLUDE_DIRS "")
set(SUPERLU_MT_LIBRARY_DIRS "")
set(SUPERLU_MT_LIBRARY_DIRS "")

find_path(SUPERLU_MT_INCLUDE_DIR
    NAMES supermatrix.h slu_mt_util.h pssp_defs.h pdsp_defs.h pcsp_defs.h pzsp_defs.h
    HINTS ${SUPERLU_MT_ROOT} ${CMAKE_INSTALL_PREFIX}/include
)

find_library(SUPERLU_MT_LIBRARY
    NAMES superlu_mt_PTHREAD superlu_mt superlu_mt_OPENMP superlumt
    PATHS "${SUPERLU_MT_INCLUDE_DIR}/../lib/"
)

message(WARNING "${SUPERLU_MT_LIBRARY}")

if(SUPERLU_MT_INCLUDE_DIR AND SUPERLU_MT_LIBRARY)
    set(SUPERLU_MT_FOUND TRUE)
    set(SUPERLU_MT_INCLUDE_DIRS ${SUPERLU_MT_INCLUDE_DIR})
    set(SUPERLU_MT_LIBRARIES ${SUPERLU_MT_LIBRARY})
    set(SUPERLU_MT_LIBRARY_DIRS ${CMAKE_INSTALL_PREFIX}/lib)

    add_library(superlu_mt::superlu_mt INTERFACE IMPORTED)
    set_target_properties(superlu_mt::superlu_mt PROPERTIES
        INTERFACE_LINK_LIBRARIES "${SUPERLU_MT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SUPERLU_MT_INCLUDE_DIRS}"
    )
endif()

if(SUPERLU_MT_FOUND)
    message(STATUS "Found SuperLU-MT: ${SUPERLU_MT_LIBRARIES}")
else()
    message(WARNING "SuperLU-MT not found.")
endif()

set(SUPERLU_MT_INCLUDE_DIRS ${SUPERLU_MT_INCLUDE_DIRS} CACHE PATH "SuperLU-MT include directories")
set(SUPERLU_MT_LIBRARIES ${SUPERLU_MT_LIBRARIES} CACHE PATH "SuperLU-MT libraries")
set(SUPERLU_MT_FOUND ${SUPERLU_MT_FOUND} CACHE BOOL "Is SuperLU-MT found?")
