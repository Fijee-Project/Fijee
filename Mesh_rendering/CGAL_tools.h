#ifndef CGAL_TOOLS_H_
#define CGAL_TOOLS_H_
//
// CGAL
//
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Labeled_image_mesh_domain_3.h>
#include <CGAL/Image_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/centroid.h>
// K-nearest neighbor algorithm with CGAL
#include <CGAL/basic.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
//
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Labeled_image_mesh_domain_3<CGAL::Image_3,Kernel> Mesh_domain;
typedef CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
typedef typename C3t3::Triangulation Triangulation;
typedef typename C3t3::Cells_in_complex_iterator Cell_iterator;
typedef typename Triangulation::Vertex_handle Vertex_handle;
typedef typename Triangulation::Cell_handle Cell_handle;
typedef typename Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
typedef typename Triangulation::Point Point_3;
//
// K-nearest neighbor algorithm with CGAL
typedef CGAL::Search_traits_3< Kernel >Traits_base;

namespace Domains
{
  // -----------------------------------
  // K-nearest neighbor algorithm (CGAL)
  // Conductivity_matching_knn() func
  // -----------------------------------
  struct Position_property_map
  {
    typedef Point_3 value_type;
    typedef const value_type& reference;
    typedef const std::tuple< Point_3, int >& key_type;
    typedef boost::readable_property_map_tag category;
  };
  // get function for the property map
  Position_property_map::reference 
    get( Position_property_map, Position_property_map::key_type );
  //
  typedef CGAL::Search_traits_adapter< std::tuple< Point_3, int >, Position_property_map, Traits_base > Traits;
  typedef CGAL::Orthogonal_k_neighbor_search< Traits > K_neighbor_search;
  typedef K_neighbor_search::Tree Tree;

  // -----------------------------------
  // Rebin_cell_pmap
  // -----------------------------------
  template <typename C3T3>
    class Rebind_cell_pmap
    {
      typedef typename C3T3::Subdomain_index Subdomain_index;
      typedef std::map<Subdomain_index,int> Subdomain_map;
      typedef typename C3T3::Cell_handle Cell_handle;
      typedef unsigned int size_type;

    public:
    Rebind_cell_pmap(const C3T3& c3t3)
      : r_c3t3_(c3t3)
      {
	typedef typename C3T3::Cells_in_complex_iterator Cell_iterator;
	
	int first_index = 0;
	int index_counter = first_index + 1;
	
	for( Cell_iterator cell_it = r_c3t3_.cells_in_complex_begin();
	     cell_it != r_c3t3_.cells_in_complex_end();
	     ++cell_it)
	  {
	    // Add subdomain index in internal map if needed
	    if ( subdomain_map_.end() ==
		 subdomain_map_.find(r_c3t3_.subdomain_index(cell_it)) )
	      {
		subdomain_map_.insert(std::make_pair(r_c3t3_.subdomain_index(cell_it),
						     index_counter));
		++index_counter;
	      }
	  }
	
	// Rebind indices in alphanumeric order
	index_counter = first_index + 1;
	for ( typename Subdomain_map::iterator mit = subdomain_map_.begin() ;
	      mit != subdomain_map_.end() ;
	      ++mit )
	  {
	    mit->second = index_counter++;
	  }
	
#ifdef CGAL_MESH_3_IO_VERBOSE
	std::cerr << "Nb of subdomains: " << subdomain_map_.size() << "\n";
	std::cerr << "Subdomain mapping:\n\t" ;
	
	typedef typename Subdomain_map::iterator Subdomain_map_iterator;
	for ( Subdomain_map_iterator sub_it = subdomain_map_.begin() ;
	      sub_it != subdomain_map_.end() ;
	      ++sub_it )
	  {
	    std::cerr << "[" << (*sub_it).first << ":" << (*sub_it).second << "] ";
	  }
	std::cerr << "\n";
#endif
      }
      
      int subdomain_index(const Cell_handle& ch) const
      {
	//	return subdomain_index(r_c3t3_.subdomain_index(ch));
	return r_c3t3_.subdomain_index(ch);
      }
      
      size_type subdomain_number() const
      {
	return subdomain_map_.size();
      }
      
    private:
      int subdomain_index(const Subdomain_index& index) const
      {
	typedef typename Subdomain_map::const_iterator Smi;
	Smi elt_it = subdomain_map_.find(index);
	if ( elt_it != subdomain_map_.end() )
	  return elt_it->second;
	else
	  return -1;
      }
      
    private:
      const C3T3& r_c3t3_;
      Subdomain_map subdomain_map_;
    };
  typedef Rebind_cell_pmap<C3t3> Cell_pmap;
  
  // -----------------------------------
  // No_patch_facet_pmap_first
  // -----------------------------------
  template <typename C3T3, typename Cell_pmap>
    class No_patch_facet_pmap_first
  {
    typedef typename C3T3::Surface_patch_index Surface_patch_index;
    typedef typename C3T3::Facet Facet;
    typedef typename C3T3::Cell_handle Cell_handle;
    
  public:
  No_patch_facet_pmap_first(const C3T3&, const Cell_pmap& cell_pmap)
    : cell_pmap_(cell_pmap) { }
    
    int surface_index(const Facet& f) const
    {
      Cell_handle c1 = f.first;
      Cell_handle c2 = c1->neighbor(f.second);
      
//      int label1 = get(cell_pmap_,c1); 
//      int label2 = get(cell_pmap_,c2);
      int label1 = cell_pmap_.subdomain_index( c1 ); 
      int label2 = cell_pmap_.subdomain_index( c2 );
      
      if ( 0 == label1 || -1 == label1 )
	label1 = label2;
      if ( 0 == label2 || -1 == label2 )
	label2 = label1;
      
      return (std::min)(label1,label2);
    }
    
  private:
    const Cell_pmap& cell_pmap_;
  };
  typedef No_patch_facet_pmap_first<C3t3,Cell_pmap> Facet_pmap;
  
  // -----------------------------------
  // Default_vertex_index_pmap
  // ----------------------------------- 
  template <typename C3T3, typename Cell_pmap, typename Facet_pmap>
    class Default_vertex_pmap
  {
    typedef typename C3T3::Surface_patch_index Surface_patch_index;
    typedef typename C3T3::Subdomain_index Subdomain_index;
    typedef typename C3T3::Index Index;
    typedef typename C3T3::Vertex_handle Vertex_handle;
    typedef typename C3T3::Cell_handle Cell_handle;
    typedef typename C3T3::Facet Facet;
    
  public:
  Default_vertex_pmap(const C3T3& c3t3,
                      const Cell_pmap& c_pmap,
                      const Facet_pmap& f_pmap)
    : c_pmap_(c_pmap), 
      f_pmap_(f_pmap), 
      r_c3t3_(c3t3), 
      edge_index_(0) {}
    
    int index(const Vertex_handle& vh) const
    {
      switch ( r_c3t3_.in_dimension(vh) )
	{
	case 2:
	  {
	    // Check if each incident surface facet of vh has the same surface index
	    typename std::vector<Facet> facets;
	    r_c3t3_.triangulation().finite_incident_facets(
							   vh, std::back_inserter(facets));
	    
	    if ( facets.begin() == facets.end() )
	      return -1;

	    // Look for the first surface facet
	    typename std::vector<Facet>::iterator it_facet = facets.begin();
	    while ( ! r_c3t3_.is_in_complex(*it_facet) )
	      {
		if ( ++it_facet == facets.end() )
		  return -1;
	      }

	    Surface_patch_index facet_index = r_c3t3_.surface_patch_index(*it_facet);
	    Facet facet = *it_facet;
	    ++it_facet;

	    for( ; it_facet != facets.end() ; ++it_facet)
	      {
		// If another index is found, return value for edge vertice
		if (   r_c3t3_.is_in_complex(*it_facet)
		       && facet_index != r_c3t3_.surface_patch_index(*it_facet) )
		  return edge_index_;
	      }

	    //	    return get(f_pmap_,facet);
	    return f_pmap_.surface_index( facet );
	  }
	  break;

	case 3:
	  {
	    // Returns value of any incident cell
	    typename std::vector<Cell_handle> cells;
	    r_c3t3_.triangulation().finite_incident_cells(
							  vh,std::back_inserter(cells));

	    if ( cells.begin() != cells.end() )
	      //	      return get(c_pmap_, *cells.begin());
	      return c_pmap_.subdomain_index( *cells.begin() );
	    else
	      return -1;
	  }
	  break;

	default:
	  // should not happen
	  return -1;
	  break;
	}
    }

  private:
    const Cell_pmap& c_pmap_;
    const Facet_pmap& f_pmap_;
    const C3T3& r_c3t3_;
    const unsigned int edge_index_;
  };
  typedef Default_vertex_pmap<C3t3, Cell_pmap, Facet_pmap> Vertex_pmap;
}
#endif
