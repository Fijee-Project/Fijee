//  Copyright (c) 2014, Yann Cobigo 
//  All rights reserved.     
//   
//  Redistribution and use in source and binary forms, with or without       
//  modification, are permitted provided that the following conditions are met:   
//   
//  1. Redistributions of source code must retain the above copyright notice, this   
//     list of conditions and the following disclaimer.    
//  2. Redistributions in binary form must reproduce the above copyright notice,   
//     this list of conditions and the following disclaimer in the documentation   
//     and/or other materials provided with the distribution.   
//   
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;   
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND   
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   
//     
//  The views and conclusions contained in the software and documentation are those   
//  of the authors and should not be interpreted as representing official policies,    
//  either expressed or implied, of the FreeBSD Project.  
#include "Build_dipoles_list_knn.h"
//
// We give a comprehensive type name
//
typedef Domains::Build_dipoles_list_knn DBdlknn;
typedef Domains::Access_parameters DAp;
//
// get function for the property map
//
Domains::Point_vector_property_map::reference 
Domains::get( Domains::Point_vector_property_map, Domains::Point_vector_property_map::key_type p)
{
  return std::get<0>(p);
};
//
//
//
DBdlknn::Build_dipoles_list_knn()
{
  //
  // Get the white and gray matter vertices matching tuples.
  (DAp::get_instance())->get_lh_match_wm_gm_( lh_match_wm_gm_ );
  (DAp::get_instance())->get_rh_match_wm_gm_( rh_match_wm_gm_ );
}
//
//
//
DBdlknn::Build_dipoles_list_knn( const DBdlknn& that )
{
}
//
//
//
DBdlknn::~Build_dipoles_list_knn()
{
  /* Do nothing */
}
//
//
//
void 
DBdlknn::Make_list( const std::list< Cell_conductivity >& List_cell_conductivity )
{
  // 
  // Time log
  FIJEE_TIME_PROFILER("Domain::Build_dipoles_list_knn::Make_list");

  //
  // We create the knn trees for the right and left hemispheres
  Dipoles_tree 
    lh_tree,
    rh_tree;
  //
  for( auto left_hemisphere : lh_match_wm_gm_ )
    lh_tree.insert( std::make_tuple( Dipole_position(std::get<0>(left_hemisphere).x(),
						     std::get<0>(left_hemisphere).y(),
						     std::get<0>(left_hemisphere).z()),
				     left_hemisphere) );
  //
  for( auto right_hemisphere : rh_match_wm_gm_ )
    rh_tree.insert( std::make_tuple( Dipole_position(std::get<0>(right_hemisphere).x(),
						     std::get<0>(right_hemisphere).y(),
						     std::get<0>(right_hemisphere).z()),
				     right_hemisphere) );
  //
  // Check the distance between two dipoles
  Distance_tree sparse_dipole_distribution;
  // insert a first fake value for initialisation
  sparse_dipole_distribution.insert( Dipole_position(256,256,256) );

  //
  //
  for ( auto cell_conductivity : List_cell_conductivity )
    {
      //
      // 
      if ( cell_conductivity.get_cell_subdomain_() == LEFT_GRAY_MATTER  ||
	   cell_conductivity.get_cell_subdomain_() == RIGHT_GRAY_MATTER ||
	   cell_conductivity.get_cell_subdomain_() == GRAY_MATTER )
	{
	  //
	  //
	  Point_vector cell_centroid = ( cell_conductivity.get_centroid_lambda_() )[0];
	  Dipole_position CGAL_cell_centroid( cell_centroid.x(),
					      cell_centroid.y(), 
					      cell_centroid.z() );
	  // Check nearest dipole neighbor
	  incremental_search search_knn( sparse_dipole_distribution, 
					 CGAL_cell_centroid, 1 );
//	  //
//	  std::cout << "Point: " << CGAL_cell_centroid << std::endl;
//	  std::cout << (search_knn.begin())->first << " "
//		    << std::sqrt((search_knn.begin())->second) 
//		    << std::endl;
//	  //
//	  sparse_dipole_distribution.insert(CGAL_cell_centroid);

	  //
	  //
//	  if ( (search_knn.begin())->second > .5 )
//	    {
//
//
//	      std::cout << "Point: " << CGAL_cell_centroid << std::endl;
//	      std::cout << (search_knn.begin())->first << " "
//			<< std::sqrt((search_knn.begin())->second) 
//			<< std::endl;
//	      //
//	      // we accept the dipole
//	      sparse_dipole_distribution.insert( CGAL_cell_centroid );


	  //
	  // left hemisphere
	  if( cell_centroid.x() < 0 )
	    {
	      // Record the tuple for the centroid
	      if ( Select_dipole(lh_tree, cell_centroid) )
		{
		  // if the dipole is distant enough from other dipoles
		  incremental_search search_knn( sparse_dipole_distribution, 
						 CGAL_cell_centroid, 1 );
		  // two dipoles must have a distance higher than 1 mm
		  if ( (search_knn.begin())->second > 1. )
		    {
		      // Add in the checking distance tree
		      sparse_dipole_distribution.insert( CGAL_cell_centroid );
		      // Add the dipole in the list
		      dipoles_list_.push_back( Domains::Dipole(cell_conductivity) );
		    }
		}
	    }
	  // right hemisphere
	  else
	    {
	      // Record the tuple for the centroid
	      if ( Select_dipole(rh_tree, cell_centroid) )
		{
		  // if the dipole is distant enough from other dipoles
		  incremental_search search_knn( sparse_dipole_distribution, 
						 CGAL_cell_centroid, 1 );
		  //  two dipoles must have a distance higher than 1 mm
		  if( (search_knn.begin())->second > 1. )
		    {
		      // Add in the checking distance tree
		      sparse_dipole_distribution.insert( CGAL_cell_centroid );
		      // Add the dipole in the list
		      dipoles_list_.push_back( Domains::Dipole(cell_conductivity) );
		    }
		}
	    }
	}
    }
  
  //
  //
  Make_analysis();
}
//
//
//
bool 
DBdlknn::Select_dipole( Dipoles_tree& Tree, Domains::Point_vector& Centroid )
{
  //
  // Boolean to record the dipole if we match the requierments
  bool record = false;
  // Selected white/gray matter vertices tuple the closest from the centroid
  std::tuple< Domains::Point_vector, Domains::Point_vector > wg_min;

  //
  // search the closest white-gray matter vertices to the centroid
  Dipoles_neighbor_search 
    dipoles_neighbor( Tree, 
		      Dipole_position(Centroid.x(), 
				      Centroid.y(), 
				      Centroid.z()),
		      10 );
  // distance between dipole and white matter
  Dipole_distance tr_dist;
	      
  //
  //
  for( auto filter_nearest : dipoles_neighbor )
    {
      //
      // w: white matter vertex
      // g: gray matter vertex
      // c: centroid
      Point_vector
	vertex_white( std::get<0>(std::get<1>(filter_nearest.first)) ),
	vertex_gray(  std::get<1>(std::get<1>(filter_nearest.first)) );
      // create temporary vectors
      // wg: | white matter, gray matter >
      // wc: | white matter, centroid >
      Point_vector
	wg(0.,0.,0.,
	   vertex_gray.x() - vertex_white.x(),
	   vertex_gray.y() - vertex_white.y(),
	   vertex_gray.z() - vertex_white.z()),
	wc(0.,0.,0.,
	   Centroid.x() - vertex_white.x(),
	   Centroid.y() - vertex_white.y(),
	   Centroid.z() - vertex_white.z());

      //
      // search the smallest triangle surface wcg (white, centroid, gray) within 30° angle 
      float 
	surface = 0.,
	min_surface = 10000;
      // if wg and wc are almost alined
      if( wg.cosine_theta(wc) > 0.86 &&  
	  wg.get_norm_() > tr_dist.inverse_of_transformed_distance(filter_nearest.second) )
	{
	  // triangle wcg surface
	  surface = /* 1/2 */( wc.cross(wg) ).get_norm_();
	  //
	  if( surface < min_surface &&  
	      vertex_white.cosine_theta( Centroid ) > 0.86 )
	    {
	      min_surface = surface;
	      wg_min = std::get<1>(filter_nearest.first);
	      record = true;
	    }
	}
    } // end of for( auto filter_nearest : dipoles_neighbor )

      //
      // Output for R analysis
#ifdef TRACE
#if TRACE == 100
  //
  // Record the tuple for the centroid
  if ( record )
    match_centroid_wm_gm_.push_back( std::make_tuple( Centroid, wg_min) );
#endif
#endif 
  
  //
  //
  return record;
}
//
//
//
void 
DBdlknn::Make_analysis()
{
#ifdef TRACE
#if TRACE == 100
  //
  //
  output_stream_
    << "Cent_X Cent_Y Cent_Z Cent_vX Cent_vY Cent_vZ "
    << "wm_X wm_Y wm_Z wm_vX wm_vY wm_vZ "
    << "gm_X gm_Y gm_Z gm_vX gm_vY gm_vZ "
    << "theta_cw dist_cw theta_cg dist_cg "
    << std::endl;
    
  //
  //
  for( auto cent_wm_gm : match_centroid_wm_gm_ )
    {
      //
      // "Cent_X Cent_Y Cent_Z Cent_vX Cent_vY Cent_vZ "
      output_stream_
	<< (std::get<0>(cent_wm_gm)).x() << " "
	<< (std::get<0>(cent_wm_gm)).y() << " "
	<< (std::get<0>(cent_wm_gm)).z() << " "
	<< (std::get<0>(cent_wm_gm)).vx() << " "
	<< (std::get<0>(cent_wm_gm)).vy() << " "
	<< (std::get<0>(cent_wm_gm)).vz() << " "
      // "wm_X wm_Y wm_Z wm_vX wm_vY wm_vZ "
	<< (std::get<0>(std::get<1>(cent_wm_gm))).x() << " "
	<< (std::get<0>(std::get<1>(cent_wm_gm))).y() << " "
	<< (std::get<0>(std::get<1>(cent_wm_gm))).z() << " "
	<< (std::get<0>(std::get<1>(cent_wm_gm))).vx() << " "
	<< (std::get<0>(std::get<1>(cent_wm_gm))).vy() << " "
	<< (std::get<0>(std::get<1>(cent_wm_gm))).vz() << " "	
      // "gm_X gm_Y gm_Z gm_vX gm_vY gm_vZ "
	<< (std::get<1>(std::get<1>(cent_wm_gm))).x() << " "
	<< (std::get<1>(std::get<1>(cent_wm_gm))).y() << " "
	<< (std::get<1>(std::get<1>(cent_wm_gm))).z() << " "
	<< (std::get<1>(std::get<1>(cent_wm_gm))).vx() << " "
	<< (std::get<1>(std::get<1>(cent_wm_gm))).vy() << " "
	<< (std::get<1>(std::get<1>(cent_wm_gm))).vz() << " ";	
      
      //
      // "theta_cw dist_cw theta_cg dist_cg ";
      Distance distance;
      output_stream_
	<< (std::get<0>(cent_wm_gm)).dot((std::get<0>(std::get<1>(cent_wm_gm)))) << " "
	<< std::sqrt(distance.transformed_distance(std::get<0>(cent_wm_gm),
						   (std::get<0>(std::get<1>(cent_wm_gm))))) 
	<< " "
	<< (std::get<0>(cent_wm_gm)).dot((std::get<1>(std::get<1>(cent_wm_gm)))) << " "
	<< std::sqrt(distance.transformed_distance(std::get<0>(cent_wm_gm),
						   (std::get<1>(std::get<1>(cent_wm_gm)))))
	<< std::endl;
    } //end of for( auto cent_wm_gm : match_centroid_wm_gm_ )

  //
  //
  Make_output_file("Dipole.distribution.frame");
#endif
#endif      
}
//
//
//
void
DBdlknn::Output_dipoles_list_xml()
{
  //
  // Output xml files. 
  std::string dipoles_XML = 
    (Domains::Access_parameters::get_instance())->get_files_path_output_();
  dipoles_XML += std::string("dipoles.xml");
  //
  std::ofstream dipoles_file( dipoles_XML.c_str() );
  
  //
  //
  Build_stream(dipoles_list_, dipoles_file);

  //
  //
  dipoles_file.close();
}
//
//
//
DBdlknn& 
DBdlknn::operator = ( const DBdlknn& that )
{

  //
  //
  return *this;
}
//
//
//
void
DBdlknn::Build_stream( const std::list< Domains::Dipole >& List, std::ofstream& stream )
{
  //
  //
  stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
	 << "<fijee xmlns:fijee=\"http://www.fenicsproject.org\">\n";
  
  //
  //
  stream << "  <dipoles size=\"" << List.size() << "\">\n";
  
  //
  //
  int index = 0;
  for ( auto dipole : List )
    stream << "    <dipole index=\"" << index++ << "\" " << dipole << "/>\n";
  
  //
  //
  stream << "  </dipoles>\n" 
	 << "</fijee>\n"; 
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DBdlknn& that)
{

  //
  //
  return stream;
};
