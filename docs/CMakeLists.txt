#--------------------------------------------------------------------------
# Directory Setup
#--------------------------------------------------------------------------
set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(DOXYGEN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/doxygen)
set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/src)
set(DOXYGEN_OUTPUT_DIR ${SPHINX_SOURCE}/doxygen)
set(DOXYGEN_HTML_OUTPUT_DIR ${DOXYGEN_OUTPUT_DIR}/html)

# Create output directory if it doesn't exist
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

#--------------------------------------------------------------------------
# Doxygen Configuration
#--------------------------------------------------------------------------
find_package(Doxygen REQUIRED)

# Collect source files and set up Doxygen paths
file(GLOB_RECURSE OPENTURBINE_PUBLIC_SOURCE
  ${DOXYGEN_INPUT_DIR}/*.hpp
  ${DOXYGEN_INPUT_DIR}/*.cpp
)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_HTML_OUTPUT_DIR}/index.html)
set(DOXYGEN_TAG_FILE ${DOXYGEN_HTML_OUTPUT_DIR}/tagfile.xml)
set(DOXYGEN_AWESOME_STYLE_FILE ${CMAKE_SOURCE_DIR}/submods/doxygen-awesome-css/doxygen-awesome.css)
set(DOXYFILE_IN ${DOXYGEN_SOURCE_DIR}/Doxyfile)
set(DOXYFILE_OUT ${DOXYGEN_OUTPUT_DIR}/Doxyfile_output)

# Configure Doxyfile
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Doxygen build command
add_custom_command(
  OUTPUT ${DOXYGEN_INDEX_FILE}
  DEPENDS ${OPENTURBINE_PUBLIC_SOURCE}
  DEPENDS ${DOXYGEN_SOURCE_DIR}/mainpage.md
  COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
  MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
  COMMENT "Generating API documentation with Doxygen"
)

add_custom_target(doxygen
  DEPENDS ${DOXYGEN_INDEX_FILE}
  COMMENT "Generating API documentation with Doxygen"
)

#--------------------------------------------------------------------------
# Doxysphinx Configuration
#--------------------------------------------------------------------------
find_package(Doxysphinx REQUIRED)

add_custom_target(doxysphinx
  COMMAND ${DOXYSPHINX_EXECUTABLE} build
  ${SPHINX_SOURCE} ${SPHINX_BUILD}/html ${DOXYGEN_HTML_OUTPUT_DIR}
  COMMENT "Converting Doxygen documentation to Sphinx format"
)
add_dependencies(doxysphinx doxygen)

#--------------------------------------------------------------------------
# Sphinx Configuration
#--------------------------------------------------------------------------
find_package(Sphinx REQUIRED)

# Spell checking target
# Spell check target is disabled
# add_custom_target(sphinx-spelling
#   COMMAND ${SPHINX_EXECUTABLE} -b spelling
#   ${SPHINX_SOURCE} ${SPHINX_BUILD}/spelling
#   COMMENT "Running spell check on documentation"
# )
# add_dependencies(sphinx-spelling doxysphinx)

# Main Sphinx documentation build
add_custom_target(sphinx
  COMMAND ${SPHINX_EXECUTABLE} -M html
  ${SPHINX_SOURCE} ${SPHINX_BUILD}
  -c ${SPHINX_SOURCE} -W --keep-going -n
  COMMENT "Generating HTML documentation with Sphinx"
)
# add_dependencies(sphinx sphinx-spelling)
add_dependencies(sphinx doxysphinx)

#--------------------------------------------------------------------------
# Main Documentation Target
#--------------------------------------------------------------------------
add_custom_target(docs ALL)
add_dependencies(docs sphinx)
