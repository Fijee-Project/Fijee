#
# Utils library
#
message(STATUS "Utils")

#
# Header files
set( Fijee_UTILS . Data_structure Minimizers Third_party Third_party/pugi )
foreach( DIR ${Fijee_UTILS} )
  file( GLOB HEADERS_ ${DIR}/*.h ${DIR}/*.hpp )
  list( APPEND HEADERS ${HEADERS_})
  install( FILES ${HEADERS_} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Utils/${DIR}/ )
  #
  file( GLOB SOURCES_ ${DIR}/*.cxx ${DIR}/*.cpp )
  list( APPEND SOURCES ${SOURCES_} )
endforeach()

#
# Set compiler flags, include directories and library dependencies
include_directories( ${Fijee_SOURCE_DIR} ${Fijee_SOURCE_DIR}/Fijee )
# Add CXX definitions
add_definitions( ${Fijee_CXX_DEFINITIONS} )
# Add flags
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}" )
set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${Fijee_LINK_FLAGS}" )
# Define libraries
add_library( FijeeUtils ${Fijee_H} ${HEADERS} ${SOURCES} )
# Add FijeeUtils target libraries
#target_link_libraries( FijeeUtils ${Fijee_TARGET_LINK_LIBRARIES} )
# Install
install( TARGETS FijeeUtils EXPORT FijeeTargets LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
