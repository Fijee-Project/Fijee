#ifndef CGAL_IMAGE_FILTERING_H
#define CGAL_IMAGE_FILTERING_H
//
// CGAL 
//
#include <CGAL/Image_3.h>
#include <CGAL/remove_outliers.h>
#include <CGAL/property_map.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/squared_distance_3.h> 
#include <CGAL/Direction_3.h>
//
// UCSF
//
#include "Utils/Statistical_analysis.h"

//#include <Eigen/Dense>


namespace Domains
{
  /**
   */
  template<class Image_,
    class BGT,
    typename word_type = unsigned char>
    class CGAL_image_filtering : public Utils::Statistical_analysis
    {
    // Types
    typedef typename BGT::Point_3   Point_3;
    typedef typename BGT::Vector_3  Vector_3;
    typedef CGAL::Search_traits_3< BGT > TreeTraits;
    typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
    typedef typename Neighbor_search::Tree Tree;
   
    private:
    //! Labeled image to wrap
    const Image_& r_im_;
    //! Position matrix
    double** positions_;
    //! Filter points
    std::vector< Point_3 > points_;
    //! Out points
    std::vector< Point_3 > out_points_;
    //! Nearest neighbor tree for filtered points
    Tree filtered_point_;
    //! Nearest neighbor tree for filtered out points
    Tree filtered_out_point_;
    
    public:
    CGAL_image_filtering(const Image_& image, double** Positions ): 
    r_im_( image ), positions_( Positions ){};
   ~CGAL_image_filtering(){/* Do nothing */}

    int operator()(const Point_3& p, const bool = true) const
    {
      return static_cast<int>( r_im_.labellized_trilinear_interpolation(CGAL::to_double(p.x()),
									CGAL::to_double(p.y()),
									CGAL::to_double(p.z()),
									word_type(0)) );
    }

    void init( const double, const double );

    void holes_detection( const double );

    bool inside(Point_3& p)
    {
      // Compute the nearest neighbor
      Neighbor_search search( filtered_point_, p, /* neraest neighbor */ 1 );
      typename Neighbor_search::iterator near = search.begin();

      //
      return ( near->second < 1.e-3 ? true : false );
    }

    bool in_hole(Point_3& p)
    {
      if( filtered_out_point_.size() == 0 )
	{
	  std::cerr << "Initialization needed with holes_detection function member" << std::endl;
	  abort();
	}
      // Compute the nearest neighbor
      Neighbor_search search( filtered_out_point_, p, /* neraest neighbor */ 1 );
      typename Neighbor_search::iterator near = search.begin();

      //
      return ( near->second < 1.e-3 ? true : false );
    }
    
    private:
    void k_means_clustering(std::vector<Point_3>& /*, int , int */);
    void filter_cluster( std::vector<Point_3>& Cluster, std::vector<Point_3>& Noise, 
			 double Removed_percentage )
    {
      
      int size_front_skull = Cluster.size();
      Noise.resize( size_front_skull );
      // remove_outliers sort clusters[x] return pointer on outliers
      auto outliers_pointer = CGAL::remove_outliers( Cluster.begin(), 
						     Cluster.end(), 
						     24 /* nb_neighbors */, 
						     Removed_percentage );
      // copy outliers in holes_to_bone
      std::copy( outliers_pointer,
		 Cluster.end(),
		 Noise.begin() );
      // erase outliers from cluster
      Cluster.erase( outliers_pointer,
			 Cluster.end() );
      //
      Cluster.shrink_to_fit();
      Noise.resize( size_front_skull - Cluster.size() );
    };

    private:
    /*!
     *  \brief 
     *
     *  This method 
     *
     */
    virtual void Make_analysis()
    {
#ifdef TRACE
#if TRACE == 100
      //
      //
      output_stream_
	<< "Cell_sub_domain "
	<< "X_cent Y_cent Z_cent  "
	<< "l1  l2  l3 l_long l_tang l_mean "
	<< "v11 v12 v13 "
	<< "v21 v22 v23 "
	<< "v31 v32 v33 \n";
      
      
//  //
//  // Main loop
//      for( auto cell_it : list_cell_conductivity_ )
//	{
//	  output_stream_
//	    << cell_it.get_cell_subdomain_() << " "
//	    << (cell_it.get_centroid_lambda_()[0]).x() << " " 
//	    << (cell_it.get_centroid_lambda_()[0]).y() << " " 
//	    << (cell_it.get_centroid_lambda_()[0]).z() << " ";
//	  //
//	  float
//	    l1 = (cell_it.get_centroid_lambda_()[0]).weight(),
//	    l2 = (cell_it.get_centroid_lambda_()[1]).weight(),
//	    l3 = (cell_it.get_centroid_lambda_()[2]).weight();
//	  //
//	  output_stream_
//	    << l1 << " " << l2 << " " << l3 << " " << l1 << " " 
//	    << (l2+l3)/2. << " " << (l1+l2+l3)/3. << " " ;
//	  //
//	  output_stream_
//	    << (cell_it.get_centroid_lambda_()[0]).vx() << " " 
//	    << (cell_it.get_centroid_lambda_()[0]).vy() << " " 
//	    << (cell_it.get_centroid_lambda_()[0]).vz() << " "
//	    << (cell_it.get_centroid_lambda_()[1]).vx() << " " 
//	    << (cell_it.get_centroid_lambda_()[1]).vy() << " " 
//	    << (cell_it.get_centroid_lambda_()[1]).vz() << " "
//	    << (cell_it.get_centroid_lambda_()[2]).vx() << " " 
//	    << (cell_it.get_centroid_lambda_()[2]).vy() << " " 
//	    << (cell_it.get_centroid_lambda_()[2]).vz() << " ";
//	  //
//	  output_stream_ << std::endl;
//	}
      
      
      //
      // 
      Make_output_file("Data_clusters_centroides.frame");
#endif
#endif      

};


    };  // end class CGAL_image_filtering


  //
  //
  //
  template< class Image_, class BGT, typename word_type>
    void CGAL_image_filtering<Image_, BGT, word_type >::init( const double Removed_percentage,
							      const double Signal_percentage )
    {
      //
      // 
      typedef typename BGT::Point_3 Point_3;
      typedef CGAL::Search_traits_3< BGT > TreeTraits;

      //
      // Amount of signal
      if( Signal_percentage > 100 || Signal_percentage < 0 )
	{
	  std::cerr << "Amount of signal percentage must be positive below 100 %" << std::endl;
	  abort();
	}
      // 
      double tolerence = 255. * ( 100. - Signal_percentage ) / 100. ;

      //
      // create a data_label_tmp private in the different
      for ( int k = 0; k < 256; k++ )
	for ( int j = 0; j < 256; j++ )
	  for ( int i = 0; i < 256; i++ )
	    {
	      //
	      int idx = i + j*256 + k*256*256;
	      //
	      int value = static_cast<int>( r_im_.labellized_trilinear_interpolation(positions_[idx][0],
										     positions_[idx][1],
										     positions_[idx][2],
										     word_type(0)) );
	      if ( value > tolerence )
		{
		  // std::cout << "val: " << value << "a" << std::endl;
		  points_.push_back(Point_3( positions_[idx][0], 
					     positions_[idx][1], 
					     positions_[idx][2] ));
		}
	      else
		{
		  out_points_.push_back(Point_3( positions_[idx][0], 
						 positions_[idx][1], 
						 positions_[idx][2] ));
		}
	    }

      //
      // 
      // Removes outliers using erase-remove idiom.
      const int nb_neighbors = 24; // considers nearest neighbor points
      if ( Removed_percentage != 0 )
	points_.erase( CGAL::remove_outliers( points_.begin(), 
					      points_.end(), 
					      nb_neighbors, Removed_percentage ),
		       points_.end() );

 
      //
      // Build the neirest neighbor tree
      for( auto point_it = points_.begin() ; point_it != points_.end() ; point_it++)
	filtered_point_.insert(*point_it);
    }
  //
  //
  //
  template< class Image_, class BGT, typename word_type>
    void CGAL_image_filtering<Image_, BGT, word_type >::holes_detection( const double Removed_percentage )
    {
      //
      //
      std::vector<Point_3> out_point_vector;

      // 
      //
      for ( auto out_point = out_points_.begin() ; 
	    out_point != out_points_.end() ;
	    out_point++ )
	{
	  // Compute the nearest neighbor in_points from the out_point
	  Neighbor_search search( filtered_point_, *out_point, /* neraest neighbors */ 60 );
	  typename Neighbor_search::iterator near = search.begin();
	  // the distance from the out_point to the nearest in_point is less than 5mm
	  if( near->second < 25 /* = 5mm x 5mm*/)
	    {
	      Point_3 nearest_point = near->first;
	      Point_3 backward_point = *out_point;
	      std::set< double > set_of_cos;
	      bool hole = false;
	      double distance = sqrt(near->second);
	      //
	      Vector_3 base_vector( (nearest_point.x() - backward_point.x())/distance,
				    (nearest_point.y() - backward_point.y())/distance,
				    (nearest_point.z() - backward_point.z())/distance );

	      //
	      //
	      for(; near != search.end(); ++near )
		{
		  double retro_distance = sqrt(near->second);
		  // all neighbors have to be in a sphere about r radius
		  if ( near->second < 60 /* radius r */ )
		    {
		      Vector_3 retro_vector( (near->first.x() - backward_point.x())/retro_distance,
					     (near->first.y() - backward_point.y())/retro_distance,
					     (near->first.z() - backward_point.z())/retro_distance );
		      //
		      set_of_cos.insert( base_vector * retro_vector );
		    }
		  
		  //
		  // 
		  if( set_of_cos.size() != 0 )
		    {
		      for( auto cos : set_of_cos )
			{
			  //std::cout << cos << std::endl;
			  if ( cos < -0.80 )
			    hole = true;
			}
		    }
		}

	      //
	      //
	      if( hole )
		out_point_vector.push_back( *out_point );
	    }
	}

      //
      // 
      k_means_clustering( out_point_vector );


//      //
//      // Remove ouliers
//      const int nb_neighbors = 24; // considers nearest neighbor points
//      if ( Removed_percentage != 0 )
//	out_point_vector.erase( CGAL::remove_outliers( out_point_vector.begin(), 
//						       out_point_vector.end(), 
//						       nb_neighbors, Removed_percentage ),
//				out_point_vector.end() );

      //
      // Build the neirest neighbor tree
      for( auto point_it = out_point_vector.begin() ; 
	   point_it != out_point_vector.end() ; 
	   point_it++ )
	filtered_out_point_.insert( *point_it );

      //
      // Reinitialize the filtered_point tree
      filtered_point_.clear();
      //
      for( auto point_it = points_.begin() ; point_it != points_.end() ; point_it++ )
	filtered_point_.insert(*point_it);
    }
  // 
  // 
  // 
  template< class Image_, class BGT, typename word_type>
    void CGAL_image_filtering<Image_, BGT, word_type >::k_means_clustering(std::vector<typename BGT::Point_3>& Spongiosa )
  {
    std::cout << "K-means clustering" << std::endl;
    //
    // 
    typedef boost::tuple<typename BGT::Point_3,int> Centroid;
    // 
    typedef typename BGT::Point_3 Point_3;
    typedef CGAL::Search_traits_3< BGT > TreeTraits;
    typedef CGAL::Search_traits_adapter< Centroid, CGAL::Nth_of_tuple_property_map< 0, Centroid >,
      TreeTraits > Traits;
    typedef CGAL::Orthogonal_k_neighbor_search<Traits> K_neighbor_search;
    typedef typename K_neighbor_search::Tree Tree;

    //
    // Clusters
    int number_of_clusters = 10;
    std::vector< Centroid > centroids( number_of_clusters );
    // front skull
    centroids[0]  = boost::make_tuple( Point_3(128., 20.,170.), 0 );
    // right top skull
    centroids[1]  = boost::make_tuple( Point_3( 70., 20.,100.), 1 );
    // left top skull
    centroids[2]  = boost::make_tuple( Point_3(170., 20.,100.), 2 );
    // center
    centroids[3]  = boost::make_tuple( Point_3(128.,128.,128.), 3 );
    // nose
    centroids[4]  = boost::make_tuple( Point_3(128.,170.,185.), 4 );
    // right cheek
    centroids[5]  = boost::make_tuple( Point_3( 70.,170.,160.), 5 );
    // left cheek
    centroids[6]  = boost::make_tuple( Point_3(190.,170.,160.), 6 );
    // jaw
    centroids[7]  = boost::make_tuple( Point_3(128.,230.,160.), 7 );
    // spine
    centroids[8]  = boost::make_tuple( Point_3(128.,230., 70.), 8 );
    // back skull
    centroids[9]  = boost::make_tuple( Point_3(128.,128., 20.), 9 );
    //    centroids[10] = boost::make_tuple( Point_3(128.,170., 50.), 10);

    //
    // 
    std::vector< int > r_i_k( Spongiosa.size() );
    // Convergence
    bool not_converged = true;
    int  iterations    = 0;

    while( not_converged )
      {
	// 
	// Build the neirest centroid tree
	Tree centroides_tree;
	for( auto centroid = centroids.begin() ; centroid != centroids.end() ; centroid++ )
	  centroides_tree.insert( *centroid );
	
	// 
	// 
	double new_centroids[number_of_clusters][3];
	double sum_r_i_k[number_of_clusters];
	bool   not_converged_centroids[number_of_clusters];
	// initialization
	for (int k = 0 ; k < number_of_clusters ; k++ )
	  {
	    new_centroids[k][0] = 0.;
	    new_centroids[k][1] = 0.;
	    new_centroids[k][2] = 0.;
	    //
	    sum_r_i_k[k] = 0.;
	    //
	    not_converged_centroids[k] = true;
	  }
	
	// 
	// 
	for ( int point = 0 ; point < Spongiosa.size() ; point++ )
	  {
	    K_neighbor_search search(centroides_tree, Spongiosa[point], 1);
	    typename K_neighbor_search::iterator nearest_cluster = search.begin();
	    //
	    int cluster_k = boost::get<1>( nearest_cluster->first );
	    r_i_k[point] = cluster_k; 
	    // 
	    new_centroids[cluster_k][0] += Spongiosa[point].x();
	    new_centroids[cluster_k][1] += Spongiosa[point].y();
	    new_centroids[cluster_k][2] += Spongiosa[point].z();
	    // 
	    sum_r_i_k[cluster_k] += 1;
	  }
	
	// 
	// 
	for (int k = 0 ; k < number_of_clusters ; k++ )
	  {
	    // new centroid
	    Point_3 mu_k;
	    // Process the new centroid
	    if( sum_r_i_k[k] != 0 )
	      mu_k = Point_3( new_centroids[k][0] / sum_r_i_k[k],
			      new_centroids[k][1] / sum_r_i_k[k],
			      new_centroids[k][2] / sum_r_i_k[k] );
	    else
	      std::cerr << "Error on the cumulation process" << std::endl;

	    //
	    // Compare the new against the old position: convergence criteria
	    if( CGAL::squared_distance(boost::get<0>(centroids[k]), mu_k) < 1.e-6 )
	      not_converged_centroids[k] = false;
	    else
	      {
		// move the position
		centroids[k] = boost::make_tuple( mu_k, k );
		// 
		not_converged_centroids[k] = true;
	      }

	    // 
	    // Check the convergence
	    not_converged &= not_converged_centroids[k];
	    iterations++;
	  }
      }

    // 
    // 
    std::cout << "K-means algorithm converged in " << iterations 
	      << " iterations. Centroides are:"
	      << std::endl;
    // 
    for (int k = 0 ; k < number_of_clusters ; k++ )
      std::cout << "k = " << k << " " << boost::get<0>(centroids[k]) << std::endl;


    //
    //
    std::vector<Point_3> clusters[number_of_clusters];
    std::vector<Point_3> holes_to_bone[number_of_clusters];
    //
    for ( int point = 0 ; point < Spongiosa.size() ; point++ )
      clusters[r_i_k[point]].push_back( Spongiosa[point] );

    // 
    // Removes outliers
    //
    Spongiosa.clear();
    //
    // front skull
    //
    filter_cluster( clusters[0], holes_to_bone[0], 23 );
    // cluster -> spongiosa
    for( auto point : clusters[0] )
      Spongiosa.push_back(point);
    // noise -> bone
    for (auto point : holes_to_bone[0] )
      points_.push_back(point);
    //
    // right top skull
    //
    filter_cluster( clusters[1], holes_to_bone[1], 26 );
    //  cluster -> spongiosa
    for( auto point : clusters[1] )
      Spongiosa.push_back(point);
    // noise -> bone
    for (auto point : holes_to_bone[1] )
      points_.push_back(point);
    //
    // left top skull
    //
    filter_cluster( clusters[2], holes_to_bone[2], 26 );
    //  cluster -> spongiosa
    for( auto point : clusters[2] )
      Spongiosa.push_back(point);
    // noise -> bone
    for (auto point : holes_to_bone[2] )
      points_.push_back(point);
    //
    // center
    // all move as bone
    //    filter_cluster( clusters[3], holes_to_bone[3],  x);
    // cluster + noise -> bone
    for (auto point : clusters[3] )
      points_.push_back(point);
    //
    // nose
    //
    filter_cluster( clusters[4], holes_to_bone[4],  23);
    // noise -> bone & remove cluster
    for (auto point : holes_to_bone[4] )
      points_.push_back(point);
    //
    // right cheek
    //
    filter_cluster( clusters[5], holes_to_bone[5],  23);
    // noise -> bone & remove cluster
    for (auto point : holes_to_bone[5] )
      points_.push_back(point);
    //
    // left cheek
    //
    filter_cluster( clusters[6], holes_to_bone[6],  23);
    // noise -> bone & remove cluster
    for (auto point : holes_to_bone[6] )
      points_.push_back(point);
    //
    // jaw
    //  all move as bone
    //    filter_cluster( clusters[7], holes_to_bone[7],  x);
    // cluster + noise -> bone
    for (auto point : clusters[7] )
      points_.push_back(point);
    //
    // spine
    //  all removed
    //    filter_cluster( clusters[8], holes_to_bone[8],  x);
    //
    // back skull
    //  all move as bone
    //    filter_cluster( clusters[9], holes_to_bone[9],  x);
    // cluster + noise -> bone
    for (auto point : clusters[9] )
      points_.push_back(point);



//    clusters[0].erase( CGAL::remove_outliers( clusters[0].begin(), 
//					      clusters[0].end(), 
//					      nb_neighbors, 23 ),
//		       clusters[0].end() );
//    clusters[3].erase( CGAL::remove_outliers( clusters[3].begin(), 
//					      clusters[3].end(), 
//					      nb_neighbors, 23 ),
//		       clusters[3].end() );
//    clusters[4].erase( CGAL::remove_outliers( clusters[4].begin(), 
//					      clusters[4].end(), 
//					      nb_neighbors, 23 ),
//		       clusters[4].end() );
//    clusters[5].erase( CGAL::remove_outliers( clusters[5].begin(), 
//					      clusters[5].end(), 
//					      nb_neighbors, 23 ),
//		       clusters[5].end() );
//    clusters[6].erase( CGAL::remove_outliers( clusters[6].begin(), 
//					      clusters[6].end(), 
//					      nb_neighbors, 23 ),
//		       clusters[6].end() );
//    clusters[7].erase( CGAL::remove_outliers( clusters[7].begin(), 
//					      clusters[7].end(), 
//					      nb_neighbors, 23 ),
//		       clusters[7].end() );
//    // Medium clusters
//    clusters[1].erase( CGAL::remove_outliers( clusters[1].begin(), 
//					      clusters[1].end(), 
//					      nb_neighbors, 26 ),
//		       clusters[1].end() );
//    clusters[2].erase( CGAL::remove_outliers( clusters[2].begin(), 
//					      clusters[2].end(), 
//					      nb_neighbors, 26 ),
//		       clusters[2].end() );
//    clusters[9].erase( CGAL::remove_outliers( clusters[9].begin(), 
//					      clusters[9].end(), 
//					      nb_neighbors, 25 ),
//		       clusters[9].end() );
   
//    std::stringstream output_stream_filtered;
//    output_stream_filtered
//      << "k X Y Z \n";
//    // 
//    for(int k = 0 ; k < number_of_clusters ; k++  )
//      {
////	clusters[k].erase( CGAL::remove_outliers( clusters[k].begin(), 
////						 clusters[k].end(), 
////						 nb_neighbors, Removed_percentage ),
////			  clusters[k].end() );
//	//
//	for ( auto point : clusters[k] )
//	  output_stream_filtered
//	    << k << " "
//	    << point << "\n";
//      }
//    // 
//    std::ofstream file_filtered;
//    file_filtered.open( "Data_clusters_filtered.frame" );
//    file_filtered << output_stream_filtered.rdbuf();
//    file_filtered.close();  
//
//
//    //
//    //
//    std::stringstream output_stream;
//    output_stream
//      << "k "
//      << "X_mu Y_mu Z_mu "
//      << "X Y Z \n";
//    //
//    for ( int point = 0 ; point < Spongiosa.size() ; point++ )
//      output_stream
//	<< r_i_k[point] << " " 
//	<< boost::get<0>(centroids[ r_i_k[point] ]) << " "
//	<< Spongiosa[point] << "\n";
//    // 
//    std::ofstream file_;
//    file_.open( "Data_clusters_centroides.frame" );
//    file_ << output_stream.rdbuf();
//    file_.close();  

  }
}
#endif
