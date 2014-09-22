#
# Fijee libraries
#

set( Fijee_H ${Fijee_SOURCE_DIR}/Fijee/fijee.h )
install( FILES ${Fijee_H} DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
#
file( GLOB HEADERS_ Fijee_*.h )
list( APPEND HEADERS ${HEADERS_})
install( FILES ${HEADERS_} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Fijee/ )
#
add_subdirectory( Utils )
add_subdirectory( Biophysics )
add_subdirectory( Mesh_rendering )
add_subdirectory( Finite_element_method_models )
