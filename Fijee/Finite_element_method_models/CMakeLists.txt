#
# fem library
#
message(STATUS "Finite_element_method_models")

#
# Production of FEniCS headers
file( GLOB UFL_SOURCES_ *.ufl )
# FFC: '-s' silence
set( UFL_COMPILER ffc )
set( UFL_FLAG -l dolfin -s )

#
#
set( UFL_HEADERS )
foreach( file_ ${UFL_SOURCES_} ) 
  get_filename_component( FILE_UFL_ ${file_} NAME )
  string( REPLACE ".ufl" ".h" UFL_HEADER_ ${FILE_UFL_} )
  add_custom_command(
    OUTPUT ${UFL_HEADER_}
    PRE_BUILD
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${FILE_UFL_} . 
    COMMAND ${UFL_COMPILER} ${UFL_FLAG} ${FILE_UFL_}
  )
  list( APPEND UFL_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${UFL_HEADER_} )
endforeach()


#
# Header files
file( GLOB HEADERS_ *.h )
list( APPEND HEADERS ${HEADERS_} ${UFL_HEADERS} )
install( FILES ${HEADERS_} ${UFL_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Finite_element_method_models )
#
file( GLOB SOURCES_ *.cxx )
list( APPEND SOURCES ${SOURCES_} )

#
# Set compiler flags, include directories and library dependencies
include_directories( ${Fijee_SOURCE_DIR}  ${Fijee_SOURCE_DIR}/Fijee ${CMAKE_CURRENT_BINARY_DIR} )
# Add CXX definitions
add_definitions( ${Fijee_CXX_DEFINITIONS} )
# Add flags
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}" )
set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${Fijee_LINK_FLAGS}" )
# Define libraries
add_library( FijeeFEM ${Fijee_H} ${HEADERS} ${UFL_HEADERS} ${SOURCES} )
#add_dependencies( FijeeFEM ufl_processing )
# Add FijeeUtils target libraries
target_link_libraries( FijeeFEM ${Fijee_TARGET_LINK_LIBRARIES} ${DOLFIN_LIBRARIES} )
# Install
install( TARGETS FijeeFEM EXPORT FijeeTargets LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
