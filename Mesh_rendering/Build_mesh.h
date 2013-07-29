#ifndef BUILD_MESH_H_
#define BUILD_MESH_H_
//
// UCSF
//
#include "enum.h"
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
//
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Labeled_image_mesh_domain_3<CGAL::Image_3,Kernel> Mesh_domain;
typedef CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
/*!
 * \file Build_mesh.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <fstream>      // std::ifstream, std::ofstream
#include <map>
/*! \namespace Domains
 * 
 * Name space for our new package
 *
 */
namespace Domains
{
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
	return subdomain_index(r_c3t3_.subdomain_index(ch));
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


  /*! \class Build_mesh
   * \brief classe Build_mesh buils head models meshes. 
   *
   *  This class build meshes over head models provided under inria image format. Sensitive parameters are provided as mesh criterias:
   *  - facet_angle: This parameter controls the shape of surface facets. Actually, it is a lower bound for the angle (in degree) of surface facets. When boundary surfaces are smooth, the termination of the meshing process is guaranteed if the angular bound is at most 30 degrees. circumscribing the surface facet and centered on the surface patch. 
   *  - facet_distance: This parameter controls the approximation error of boundary and subdivision surfaces. Actually, it is either a constant or a spatially variable scalar field. It provides an upper bound for the distance between the circumcenter of a surface facet and the center of a surface Delaunay ball of this facet.
   *  - cell_radius_edge_ratio:  This parameter controls the shape of mesh cells (but can't filter slivers). Actually, it is an upper bound for the ratio between the circumradius of a mesh tetrahedron and its shortest edge. There is a theoretical bound for this parameter: the Delaunay refinement process is guaranteed to terminate for values of cell_radius_edge_ratio bigger than 2. 
   *  - cell_size: This parameter controls the size of mesh tetrahedra. It is either a scalar or a spatially variable scalar field. It provides an upper bound on the circumradii of the mesh tetrahedra. 
   *
   * The set (facet_angle=30, facet_size=1.2, facet_distance=.8, cell_radius_edge_ratio=2., cell_size=1.8) provide an homogeneouse 3D mesh.
   * For more information the user is invited to go on www.cgal.org.
   *
   */
  class Build_mesh
  {
  private:
    //! The variable mesh hold the CGAL mesh.
    C3t3 mesh_;

  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Build_mesh
     *
     */
    Build_mesh();
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Build_mesh( const Build_mesh& ) = delete;
    /*!
     *  \brief Move Constructor
     *
     *  Constructor is a moving constructor
     *
     */
    Build_mesh( Build_mesh&& ) = delete;
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Build_mesh
     */
    virtual ~Build_mesh();
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Build_mesh
     *
     */
    Build_mesh& operator = ( const Build_mesh& ) = delete;
    /*!
     *  \brief Move Operator =
     *
     *  Move operator of the class Build_mesh
     *
     */
    Build_mesh& operator = ( Build_mesh&& ) = delete;
    /*!
     *  \brief Operator ()
     *
     *  Object function for multi-threading
     *
     */
    void operator ()( Mesh_output );

  public:
    /*!
     *  \brief Output in .mesh format
     *
     *  This method uses the MEdit format from CGAL
     *
     */
    inline void Output_mesh_format()
    {
      std::ofstream medit_file("out.mesh");
      mesh_.output_to_medit(medit_file);
    };
    /*!
     *  \brief Output in .xlm format
     *
     *  This method export the mesh in a FEniCS mesh format.
     *
     */
    void Output_FEniCS_xml();
    /*!
     *  \brief Output in .vtu format
     *
     *  This method export the mesh in a VTK unscructured format. This format can be read by paraview.
     *
     */
    void Output_VTU_xml();
    /*!
     *  \brief Output the XML match between mesh and conductivity
     *
     *  This method matches a conductivity tensor for each cell.
     *
     */
    void Output_mesh_conductivity_xml();

  public:
  };
  /*!
   *  \brief Dump values for Build_mesh
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Build_mesh& );
};
#endif