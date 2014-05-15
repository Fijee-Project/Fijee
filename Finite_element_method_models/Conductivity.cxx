#include "Conductivity.h"

typedef Solver::PDE_solver_parameters SDEsp;

/* Tensor_conductivity */
//
//
//
Solver::Tensor_conductivity::Tensor_conductivity( std::shared_ptr< const Mesh > Mesh_head ): 
  Expression(3,3)
{
  //
  //
  std::string 
    C00 = (SDEsp::get_instance())->get_files_path_output_(), 
    C01 = (SDEsp::get_instance())->get_files_path_output_(), 
    C02 = (SDEsp::get_instance())->get_files_path_output_(), 
    C11 = (SDEsp::get_instance())->get_files_path_output_(), 
    C12 = (SDEsp::get_instance())->get_files_path_output_(), 
    C22 = (SDEsp::get_instance())->get_files_path_output_(); 
  //
    C00 += "C00.xml"; 
    C01 += "C01.xml"; 
    C02 += "C02.xml"; 
    C11 += "C11.xml"; 
    C12 += "C12.xml"; 
    C22 += "C22.xml"; 
  //
  C00_ = MeshFunction< double >(Mesh_head, C00.c_str() );
  C01_ = MeshFunction< double >(Mesh_head, C01.c_str() );
  C02_ = MeshFunction< double >(Mesh_head, C02.c_str() );
  C11_ = MeshFunction< double >(Mesh_head, C11.c_str() );
  C12_ = MeshFunction< double >(Mesh_head, C12.c_str() );
  C22_ = MeshFunction< double >(Mesh_head, C22.c_str() );
}
//
//
//
void 
Solver::Tensor_conductivity::eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
{
  //
  //  const uint D = cell.topological_dimension;
  const uint cell_index = cell.index;
  // values should not be zero
  double
    C00 = /*(*/ C00_[cell_index] /*!= 0. ? C00_[cell_index] : 0.0132 )*/,
    C01 = /*(*/ C01_[cell_index] /*!= 0. ? C01_[cell_index] : 0.0132 )*/,
    C02 = /*(*/ C02_[cell_index] /*!= 0. ? C02_[cell_index] : 0.0132 )*/,
    C11 = /*(*/ C11_[cell_index] /*!= 0. ? C11_[cell_index] : 0.0132 )*/,
    C12 = /*(*/ C12_[cell_index] /*!= 0. ? C12_[cell_index] : 0.0132 )*/,
    C22 = /*(*/ C22_[cell_index] /*!= 0. ? C22_[cell_index] : 0.0132 )*/;
  //
  values[0] = C00; values[3] = C01; values[6] = C02;
  values[1] = C01; values[4] = C11; values[7] = C12;
  values[2] = C02; values[5] = C12; values[8] = C22;
}
//
//
//
void 
Solver::Tensor_conductivity::conductivity_update( const std::shared_ptr< MeshFunction<std::size_t> > Domains,
						  const Eigen::Vector3d& Simplex )
{
  //
  // 
  for ( MeshEntityIterator cell( *(Domains->mesh()), Domains->dim() );
	!cell.end(); ++cell )
    {
      //
      // Swhitch on sub-domains
      switch ( (*Domains)[*cell] )
	{
	case SPONGIOSA_SKULL:
	  {
	    //	    std::cout << "spongiosa: " << C00_[cell->index()] << std::endl;
	    C00_[cell->index()] = C00_[cell->index()] = C00_[cell->index()] = Simplex(1);
	    break;
	  }
	case OUTSIDE_SKULL/* compacta skull */:
	  {
	    // 	    std::cout << "compacta: " << C00_[cell->index()] << std::endl;
	    C00_[cell->index()] = C00_[cell->index()] = C00_[cell->index()] = Simplex(2);
	    break;
	  }
	case OUTSIDE_SCALP/* skin */:
	  {
	    // 	    std::cout << "skin: " << C00_[cell->index()] << std::endl;
	    C00_[cell->index()] = C00_[cell->index()] = C00_[cell->index()] = Simplex(0);
	    break;
	  }
	}
    }
}
//
//
//
/* Sigma_isotrope */
//
//
//
void 
Solver::Sigma_isotrope::eval(Array<double>& values, const Array<double>& x) const
{
  //
  values[0] = sigma_; values[3] = 0.0;    values[6] = 0.0;
  values[1] = 0.0;    values[4] = sigma_; values[7] = 0.0;
  values[2] = 0.0;    values[5] = 0.0;    values[8] = sigma_;
}
//
//
//
void 
Solver::Sigma_skull::eval(Array<double>& values, const Array<double>& x) const
{
  // Geometric data
  double 
    r     = sqrt(x[0]*x[0] + x[1]*x[1]),
    theta = 2. * atan( x[1]/(x[0] + r) ),
    c_th  = cos(theta),
    s_th  = sin(theta);
  
  // Conductivity value in spherical frame
  double
    sigma_r = 0.0042,
    sigma_t = 0.0420;
  
  //
  values[0]=sigma_r*c_th*c_th + sigma_t*s_th*s_th; values[2]=sigma_r*c_th*s_th - sigma_t*s_th*c_th;
  values[1]=sigma_r*c_th*s_th - sigma_t*s_th*c_th; values[3]=sigma_r*c_th*c_th + sigma_t*s_th*s_th;
}
//
//
//
