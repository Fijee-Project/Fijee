#ifndef CGAL_IMAGE_FILTERING_H
#define CGAL_IMAGE_FILTERING_H



#include <CGAL/Image_3.h>
#include <CGAL/remove_outliers.h>
#include <CGAL/property_map.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>


//typedef CGAL::Simple_cartesian<double> K;
//typedef K::Point_2 Point_d;
//typedef CGAL::Search_traits_2<K> TreeTraits;
//typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
//typedef Neighbor_search::Tree Tree;




//namespace CGAL {
//
//namespace Mesh_3 {
namespace Domains
{
  /**
   */
  template<class Image_,
    class BGT,
    typename word_type = unsigned char>
    class CGAL_image_filtering
    {
    // Types
    typedef typename BGT::Point_3   Point_3;
    typedef CGAL::Search_traits_3< BGT > TreeTraits;
    typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
    typedef typename Neighbor_search::Tree Tree;
   
    private:
    //! Labeled image to wrap
    const Image_& r_im_;
    //! Position matrix
    double** positions_;
    //! Filter point
    std::vector< Point_3 > points_;
    //! Nearest neighbor tree
    Tree filterd_point_;
    
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

    void init( const double );

    bool inside(Point_3& p)
    {
      // Compute the nearest neighbor
      Neighbor_search search( filterd_point_, p, /* neraest neighbor */ 1 );
      typename Neighbor_search::iterator near = search.begin();

      //
      return ( near->second < 1.e-3 ? true : false );
    }

    };  // end class CGAL_image_filtering


  //
  //
  //
  template< class Image_, class BGT, typename word_type>
    void CGAL_image_filtering<Image_, BGT, word_type >::init( const double Removed_percentage )
    {
      typedef typename BGT::Point_3 Point_3;
      typedef CGAL::Search_traits_3< BGT > TreeTraits;
      typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
      typedef typename Neighbor_search::Tree Tree;
      //
      // create a data_label_tmp private in the different
      for ( int k = 0; k < 256; k++ )
	for ( int j = 0; j < 256; j++ )
	  for ( int i = 0; i < 256; i++ )
	    {
	      //
	      int idx = i + j*256 + k*256*256;
	      //
	      if ( static_cast<int>( r_im_.labellized_trilinear_interpolation(positions_[idx][0],
									      positions_[idx][1],
									      positions_[idx][2],
									      word_type(0)) ) != 0 )
		{
//		  int tempo = r_im_.labellized_trilinear_interpolation(positions_[idx][0],
//								       positions_[idx][1],
//								       positions_[idx][2],
//								       word_type(0));
//		  std::cout << "val: " << tempo << std::endl;
		  points_.push_back(Point_3( positions_[idx][0], positions_[idx][1], positions_[idx][2] ));
		}
	      
	    }

      //
      //
      std::cout << "Vector size(): " << points_.size() << std::endl;

      //
      // 
      // Removes outliers using erase-remove idiom.
      const int nb_neighbors = 26; // considers 24 nearest neighbor points
      points_.erase( CGAL::remove_outliers( points_.begin(), points_.end(), nb_neighbors, Removed_percentage ),
		     points_.end() );

      std::cout << "Vector size() after: " << points_.size() << std::endl;
      //
      // Build the neirest neighbor tree
      for( auto point_it = points_.begin() ; point_it != points_.end() ; point_it++)
	filterd_point_.insert(*point_it);
    }
}



//}  // end namespace Mesh_3
//}  // end namespace CGAL
#endif
