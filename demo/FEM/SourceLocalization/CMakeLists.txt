cmake_minimum_required( VERSION 2.6.0 )

#
# Application's name 
PROJECT( SourceLocalization )

#
# Project conf
find_package( Fijee REQUIRED )
#
include_directories( ${Fijee_INCLUDE_DIRS} )
link_directories( ${Fijee_LIBRARY_DIRS} )

#
# Other libraries
find_package( PkgConfig REQUIRED)
pkg_check_modules( gsl REQUIRED gsl )
pkg_check_modules( zlib REQUIRED zlib )

#
# Other definition
add_definitions( -O3 -std=c++0x -Wno-deprecated )  

#
# Build application
add_executable( SourceLocalization MACOSX_BUNDLE SourceLocalization )
#
target_link_libraries( SourceLocalization 
  ${Fijee_LIBRARIES} 
  ${VTK_LIBRARIES}
  ${gsl_LIBRARIES} 
  ${zlib_LIBRARIES}
  CGAL CGAL_ImageIO dolfin metis 
  gmp boost_system )
