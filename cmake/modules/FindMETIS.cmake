#
# Find the METIS includes and libraries
#
# METIS is a library that implements a variety of algorithms for
# partitioning unstructured graphs, meshes, and for computing fill-reducing orderings of
# sparse matrices. It can be found at:
#
# METIS_INCLUDE_DIR - where to find metis.h
# METIS_LIBRARIES   - List of fully qualified libraries to link against.
# METIS_FOUND       - Do not attempt to use if "no" or undefined.

find_path( METIS_INCLUDE_DIR metis.h
  ${METIS_DIR}/include
  /opt/local/include
  /usr/local/include
  /usr/include
  /usr/include/metis
#  $ENV{METIS_DIR}/include
  )

find_library( METIS_LIBRARY metis
  HINTS ${METIS_DIR}/lib
  PATHS /opt/local/lib /usr/local/lib /usr/lib
#  $ENV{METIS_DIR}/lib
  )
#
get_filename_component( METIS_LIBRARY_DIR "${METIS_LIBRARY}" PATH)

if(METIS_INCLUDE_DIR)
  add_definitions( -DIDXTYPEWIDTH=64 -DINTSIZE64 )

  if( METIS_LIBRARY )
    set( METIS_FOUND "YES" )
  endif( METIS_LIBRARY )
endif( METIS_INCLUDE_DIR )

mark_as_advanced( METIS_INCLUDE_DIR METIS_LIBRARY )
