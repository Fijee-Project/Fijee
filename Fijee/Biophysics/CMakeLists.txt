#
# Biophysics library
#

#
# Header files
file( GLOB HEADERS_ *.h )
list( APPEND HEADERS ${HEADERS_})
install( FILES ${HEADERS_} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Biophysics )
#
file( GLOB SOURCES_ *.cxx )
list( APPEND SOURCES ${SOURCES_} )

#
# Set compiler flags, include directories and library dependencies
include_directories( ${Fijee_SOURCE_DIR}  ${Fijee_SOURCE_DIR}/Fijee )
# Add CXX definitions
add_definitions( ${Fijee_CXX_DEFINITIONS} )
# Add flags
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}" )
set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${Fijee_LINK_FLAGS}" )
# Define libraries
add_library( FijeeBiophysics ${Fijee_H} ${HEADERS} ${SOURCES} )
# Add FijeeUtils target libraries
target_link_libraries( FijeeBiophysics ${Fijee_TARGET_LINK_LIBRARIES} )
# Install
install( TARGETS FijeeBiophysics EXPORT FijeeTargets LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
