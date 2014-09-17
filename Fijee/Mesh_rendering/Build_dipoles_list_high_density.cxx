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
#include "Build_dipoles_list_high_density.h"
//
// We give a comprehensive type name
//
typedef Domains::Build_dipoles_list_high_density DBdlhd;
typedef Domains::Access_parameters DAp;
//
// get function for the property map
//
Domains::Point_vector_high_density_map::reference 
Domains::get( Domains::Point_vector_high_density_map, Domains::Point_vector_high_density_map::key_type p)
{
  return std::get<0>(p);
};
Domains::Point_vector_parcellation_map::reference 
Domains::get( Domains::Point_vector_parcellation_map, Domains::Point_vector_parcellation_map::key_type p)
{
  return std::get<0>(p);
};
//
//
//
DBdlhd::Build_dipoles_list_high_density():
  add_gray_matter_(false), cell_size_(3. /*mm*/), layer_(1)
{
  //
  // Get the white and gray matter point_vectors
  // 
  (DAp::get_instance())->get_lh_white_matter_surface_point_normal_( lh_wm_ );
  (DAp::get_instance())->get_rh_white_matter_surface_point_normal_( rh_wm_ );
  // 
  if( add_gray_matter_ )
    {
      (DAp::get_instance())->get_lh_gray_matter_surface_point_normal_( lh_gm_ );
      (DAp::get_instance())->get_rh_gray_matter_surface_point_normal_( rh_gm_ );
    }
}
//
//
//
DBdlhd::Build_dipoles_list_high_density( const DBdlhd& that )
{
}
//
//
//
DBdlhd::~Build_dipoles_list_high_density()
{
  /* Do nothing */
}
//
//
//
void 
DBdlhd::Make_list( const std::list< Cell_conductivity >& List_cell_conductivity )
{
  // 
  // Time log
  FIJEE_TIME_PROFILER("Domain::Build_dipoles_list_high_density::Make_list");

  // 
  // Initialize index and Point vector tuple.
  // white matter
  std::vector< IndexedPointVector > lh_wm_points( lh_wm_.size() + lh_gm_.size() );
  std::vector< IndexedPointVector > rh_wm_points( rh_wm_.size() + rh_gm_.size());
  // Concat {white,gray} matters
  lh_wm_.insert( lh_wm_.end(), lh_gm_.begin(), lh_gm_.end() );
  rh_wm_.insert( rh_wm_.end(), rh_gm_.begin(), rh_gm_.end() );

  // 
  // Left hemisphere
  int l = 0;
  // 
  for( auto vertex : lh_wm_ )
    {
      lh_wm_points[l].get<0>() = l; 
      lh_wm_points[l].get<1>() = Dipole_position( vertex.x(), vertex.y(), vertex.z() ); 
      lh_wm_points[l].get<2>() = vertex;
      // 
      l++;
    }
  // Right hemisphere
  int r = 0;
  for( auto vertex : rh_wm_ )
    {
      rh_wm_points[r].get<0>() = r; 
      rh_wm_points[r].get<1>() = Dipole_position( vertex.x(), vertex.y(), vertex.z() ); 
      rh_wm_points[r].get<2>() = vertex;
      // 
      r++;
    }
  // 
  std::cout << "Density of the gray matter 2D mesh: " 
	    << lh_wm_.size() + rh_wm_.size() + lh_gm_.size() + rh_gm_.size() 
	    << std::endl;

  // 
  // Compute the density of dipoles
  // 
  
  // 
  // Computes average spacing.
  const unsigned int nb_neighbors = 6; // 1 ring
  // 
  FT lh_average_spacing = CGAL::compute_average_spacing(lh_wm_points.begin(), lh_wm_points.end(),
							CGAL::Nth_of_tuple_property_map<1,IndexedPointVector>(),
							nb_neighbors);
  // 
  FT rh_average_spacing = CGAL::compute_average_spacing(rh_wm_points.begin(), rh_wm_points.end(),
							CGAL::Nth_of_tuple_property_map<1,IndexedPointVector>(),
							nb_neighbors);
  // 
  std::cout << "Average spacing dipole: " 
	    << lh_average_spacing << " mm in the left hemisphere, and " 
	    << rh_average_spacing << " mm in the right hemisphere" << std::endl;
  
  // 
  // Control of density
  // Left hemisphere
  lh_wm_points.erase( CGAL::grid_simplify_point_set(lh_wm_points.begin(), 
						    lh_wm_points.end(),  
						    CGAL::Nth_of_tuple_property_map< 1, IndexedPointVector >(),
						    cell_size_),
		      lh_wm_points.end() );
  // Right hemisphere
  rh_wm_points.erase( CGAL::grid_simplify_point_set(rh_wm_points.begin(), 
						    rh_wm_points.end(),  
						    CGAL::Nth_of_tuple_property_map< 1, IndexedPointVector >(),
						    cell_size_),
		      rh_wm_points.end() );
  // 
  std::cout << "Down sizing the dipole distribution d(d1,d2) > " << cell_size_ << " mm: " 
	    << lh_wm_points.size() + rh_wm_points.size() << " dipoles in the 2D mesh" 
	    << std::endl;


  //
  // Build the knn tree of mesh cell
  // 
  
  // Define two trees to reduce the complexity of the search algorithm
  High_density_tree 
    lh_tree,
    rh_tree;

  // 
  std::vector< bool > cell_conductivity_assignment(List_cell_conductivity.size(), false);
  // 
  for ( auto cell_conductivity : List_cell_conductivity )
    {
      // 
      // Centroid
      Point_vector cell_centroid = ( cell_conductivity.get_centroid_lambda_() )[0];
      // 
      switch(cell_conductivity.get_cell_subdomain_())
	{
	case GRAY_MATTER: // 4-spheres case
	  {
	    if( cell_centroid.x() < 0 )
	      lh_tree.insert(std::make_tuple(Dipole_position( cell_centroid.x(),
							      cell_centroid.y(),
							      cell_centroid.z() ),
					     cell_conductivity));
	    else // right hemisphere
	      rh_tree.insert(std::make_tuple(Dipole_position( cell_centroid.x(),
							      cell_centroid.y(),
							      cell_centroid.z() ),
					     cell_conductivity));
	    break;
	  }
	case LEFT_GRAY_MATTER:  // head left gray matter hemisphere
	  {
	    lh_tree.insert(std::make_tuple(Dipole_position( cell_centroid.x(),
							    cell_centroid.y(),
							    cell_centroid.z() ),
					   cell_conductivity));
	    break;
	  }
	case RIGHT_GRAY_MATTER: // head right gray matter hemisphere
	  {
	    rh_tree.insert(std::make_tuple(Dipole_position( cell_centroid.x(),
							    cell_centroid.y(),
							    cell_centroid.z() ),
					   cell_conductivity));
	    break;
	  }
//	default:
//	  {
//	    std::cerr << "All gray matter centroid must be associated to a subdomain: " 
//		      << "Subdomain is: " << cell_conductivity.get_cell_subdomain_()
//		      << std::endl;
//	    abort();
//	  }
	}
    }

  // 
  // Dipole list creation
  // 
  // Left hemisphere
  Select_dipole(lh_tree, lh_wm_points, cell_conductivity_assignment);
  // Right hemisphere
  Select_dipole(rh_tree, rh_wm_points, cell_conductivity_assignment);
  // 
  std::cout << "Gray matter dipole distribution: " 
	    << dipoles_list_.size() 
	    << " dipoles"
	    << std::endl;

  // 
  // Parcellation list creation
  // 
  Parcellation_list();

  //
  //
  Make_analysis();
}
//
//
//
void 
DBdlhd::Select_dipole( const High_density_tree& Tree, 
		       const std::vector< IndexedPointVector >& PV_vector,
		       std::vector< bool >& Cell_assignment )
{
  // 
  // Left hemisphere
  for( auto vertex : PV_vector )
    {
      // 
      // Build the search tree
      High_density_neighbor_search
	dipoles_neighbor( Tree, vertex.get<1>(), layer_ );

      // 
      // Position of the dipole in the nearest mesh cell centroid
      auto dipole_position = dipoles_neighbor.begin();
      // If the conductivity centroid is not yet assigned; otherwise we skip it
      for (int layer = 0 ; layer < layer_ ; layer++ )
	{
	  if( !Cell_assignment[(std::get<1>(dipole_position->first)).get_cell_id_()] )
	    {
	      // Assignes the centroid
	      Cell_assignment[(std::get<1>(dipole_position->first)).get_cell_id_()] = true;
	      // Add the dipole in the list
	      dipoles_list_.push_back( Domains::Dipole(Point_vector((std::get<0>(dipole_position->first)).x(),
								    (std::get<0>(dipole_position->first)).y(),
								    (std::get<0>(dipole_position->first)).z(),
								    (vertex.get<2>()).vx(),
								    (vertex.get<2>()).vy(),
								    (vertex.get<2>()).vz()),
						       std::get<1>(dipole_position->first)) );
	      //
	      // R study
#ifdef TRACE
#if TRACE == 100
	      centroid_vertex_.push_back(std::make_tuple(((std::get<1>(dipole_position->first)).get_centroid_lambda_())[0],
							 vertex.get<2>(), 
							 std::get<1>(dipole_position->first)));
#endif
#endif
	      break;
	    }

	  // 
	  // next layer
	  dipole_position++;
	}
    }
}
//
//
//
void 
DBdlhd::Parcellation_list()
{
  // 
  // Region vector 
  // (2*number_of_parcels + 1) because regions start at 1
  // region 0 won't be used.
  std::vector< std::list< Domains::Dipole > > 
    Regions( 2 * (DAp::get_instance())->get_number_of_parcels_() + 1 );

  // 
  // Sort the dipoles in function of the region they belong too.
  for ( auto dipole : dipoles_list_ )
    Regions[ dipole.get_cell_parcel_() ].push_back(dipole);

  // 
  // Region centroid -> dipole centroid
  // 
  for ( auto dipoles : Regions )
    if ( dipoles.size() != 0 ) // We get rid of Region 0
      {
	// 
	// In a region
	// 
	std::list< Dipole_position > positions_list;
	Parcellation_tree tree;

	// 
	// 
	for ( auto dipole : dipoles )
	  {
	    // 
	    // Create the position list and the neirest neighbor tree of all dipole
	    Dipole_position position(dipole.x(), dipole.y(), dipole.z());
	    // 
	    positions_list.push_back(position);
	    tree.insert(std::make_tuple(position,dipole));
	  }

	// 
	// Build the search tree
	Parcellation_neighbor_search
	  dipoles_neighbor( tree, 
			    CGAL::centroid(positions_list.begin(), 
					   positions_list.end()), 
			    1 );
	// Save the dipole the closest from the centroid
	parcellation_list_.push_back( std::get<1>((dipoles_neighbor.begin())->first) );
      }
}
//
//
//
void 
DBdlhd::Make_analysis()
{
#ifdef TRACE
#if TRACE == 100
  //
  //
  output_stream_
    // Dipole Region
    << "Region " 
    // Dipole point-vector
    << "Dipole_X Dipole_Y Dipole_Z Dipole_vX Dipole_vY Dipole_vZ " 
    // Mesh centroide point-vector
    << "Cent_X Cent_Y Cent_Z Cent_vX Cent_vY Cent_vZ "
    // 2D mesh point-vector
    << "Vertex_X Vertex_Y Vertex_Z Vertex_vX Vertex_vY Vertex_vZ "
    << std::endl;
    
  //
  //
  for( auto centroid_vertex : centroid_vertex_ )
    {
      //
      // 
      output_stream_
	// "Region "
	<< ((std::get<2>(centroid_vertex))).get_cell_parcel_() << " "
	// "Dipole_X Dipole_Y Dipole_Z Dipole_vX Dipole_vY Dipole_vZ "
	<< ((std::get<0>(centroid_vertex))).x() << " "
	<< ((std::get<0>(centroid_vertex))).y() << " "
	<< ((std::get<0>(centroid_vertex))).z() << " "
	<< ((std::get<1>(centroid_vertex))).vx() << " "
	<< ((std::get<1>(centroid_vertex))).vy() << " "
	<< ((std::get<1>(centroid_vertex))).vz() << " "
      // "Cent_X Cent_Y Cent_Z Cent_vX Cent_vY Cent_vZ "
	<< (std::get<0>(centroid_vertex)).x() << " "
	<< (std::get<0>(centroid_vertex)).y() << " "
	<< (std::get<0>(centroid_vertex)).z() << " "
	<< (std::get<0>(centroid_vertex)).vx() << " "
	<< (std::get<0>(centroid_vertex)).vy() << " "
	<< (std::get<0>(centroid_vertex)).vz() << " "	
      // "Vertex_X Vertex_Y Vertex_Z Vertex_vX Vertex_vY Vertex_vZ "
	<< ((std::get<1>(centroid_vertex))).x() << " "
	<< ((std::get<1>(centroid_vertex))).y() << " "
	<< ((std::get<1>(centroid_vertex))).z() << " "
	<< ((std::get<1>(centroid_vertex))).vx() << " "
	<< ((std::get<1>(centroid_vertex))).vy() << " "
	<< ((std::get<1>(centroid_vertex))).vz() << " "
	<< std::endl;	
    } 

  //
  //
  Make_output_file("Dipole_high_density.distribution.frame");

  // 
  // 
    output_stream_
    // Dipole Region
    << "Region " 
    // Dipole point-vector
    << "Dipole_X Dipole_Y Dipole_Z Dipole_vX Dipole_vY Dipole_vZ " 
    << std::endl;
    
  //
  //
  for( auto dipole  : parcellation_list_ )
    output_stream_
      // Dipole Region
      << dipole.get_cell_parcel_() << " "
      // Dipole point-vector
      << dipole.x() << " "
      << dipole.y() << " "
      << dipole.z() << " "
      << dipole.vx() << " "
      << dipole.vy() << " "
      << dipole.vz() << " "
      << std::endl;

  //
  //
  Make_output_file("Parcellation.frame");

#endif
#endif      
}
//
//
//
void
DBdlhd::Output_dipoles_list_xml()
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
void
DBdlhd::Output_parcellation_list_xml()
{
  //
  // Output xml files. 
  std::string dipoles_XML = 
    (Domains::Access_parameters::get_instance())->get_files_path_output_();
  dipoles_XML += std::string("parcellation.xml");
  //
  std::ofstream dipoles_file( dipoles_XML.c_str() );
  
  //
  //
  Build_stream(parcellation_list_, dipoles_file);

  //
  //
  dipoles_file.close();
}
//
//
//
DBdlhd& 
DBdlhd::operator = ( const DBdlhd& that )
{

  //
  //
  return *this;
}
//
//
//
void
DBdlhd::Build_stream( const std::list< Domains::Dipole >& List, std::ofstream& stream )
{
  //
  //
  stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
	 << "<fijee xmlns:fijee=\"https://github.com/Fijee-Project/Fijee\">\n";
  
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
		       const DBdlhd& that)
{

  //
  //
  return stream;
};
