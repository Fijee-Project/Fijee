#
# Biophysics library
#

#
# Header files
file( GLOB HEADERS_ *.h )
list( APPEND HEADERS ${HEADERS_})
install( FILES ${HEADERS_} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Mesh_rendering )
#
file( GLOB SOURCES_ *.cxx )
list( APPEND SOURCES ${SOURCES_} )

#
# CUDA
set( CUDA_SOURCES_ "" )
if( Fijee_CUDA )
  file( GLOB CUDA_SOURCES_ *.cu )
  cuda_compile( CUDA_OBJ_ ${HEADERS} ${CUDA_SOURCES_} MODULE )
endif()


#
# Set compiler flags, include directories and library dependencies
include_directories( ${Fijee_SOURCE_DIR}  ${Fijee_SOURCE_DIR}/Fijee )
# Add CXX definitions
add_definitions( ${Fijee_CXX_DEFINITIONS} )
# Add flags
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}" )
set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${Fijee_LINK_FLAGS}" )
# Define libraries
add_library( FijeeMesh ${Fijee_H} ${HEADERS} ${SOURCES} ${CUDA_SOURCES_} )
#add_dependencies( FijeeMesh FijeeUtils )
# Add FijeeUtils target libraries
target_link_libraries( FijeeMesh ${VTK_LIBRARIES} ${CUDA_LIBRARIES} )
# Install
install( TARGETS FijeeMesh EXPORT FijeeTargets LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
