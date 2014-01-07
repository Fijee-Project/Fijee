#include "Dipole.h"
//
// We give a comprehensive type name
//
typedef Domains::Dipole DD;
//
//
//
DD::Dipole():
  Domains::Point_vector(),
  cell_id_(0), cell_subdomain_(0)
{
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = 0.;
  //
  for( int i = 0 ; i < 3 ; i++)
    conductivity_eigenvalues_[i] = 0.;
}
//
//
//
DD::Dipole( const DD& that ):
  Domains::Point_vector(that),
  cell_id_(that.cell_id_), cell_subdomain_(that.cell_subdomain_)
{
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = that.conductivity_coefficients_[i];
  //
  for( int i = 0 ; i < 3 ; i++)
    conductivity_eigenvalues_[i] = that.conductivity_eigenvalues_[i];
}
//
//
//
DD::Dipole( const Domains::Cell_conductivity& that ):
  Domains::Point_vector( that.get_centroid_lambda_()[0] ),
  cell_id_( that.get_cell_id_() ), cell_subdomain_(that.get_cell_subdomain_())
{
  //
  // set the dipole intensity
  set_weight_( 0.000000001 );
  
  //
  //
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = that.get_conductivity_coefficients_()[i];

  //
  //
  for( int i = 0 ; i < 3 ; i++)
    conductivity_eigenvalues_[i] = that.get_centroid_lambda_()[i].weight();
  
}
//
//
//
DD::~Dipole(){ /* Do nothing */}
//
//
//
DD& 
DD::operator = ( const DD& that )
{
  Domains::Point_vector::operator = (that);
  //
  cell_id_        = that.cell_id_;
  cell_subdomain_ = that.cell_subdomain_;
  //
  for( int i = 0 ; i < 6 ; i++)
    conductivity_coefficients_[i] = that.conductivity_coefficients_[i];
  //
  for( int i = 0 ; i < 3 ; i++)
    conductivity_eigenvalues_[i] = that.conductivity_eigenvalues_[i];

  //
  //
  return *this;
}
//
//
//
std::ostream& 
Domains::operator << ( std::ostream& stream, 
		       const DD& that)
{
  //
  //
  stream 
    // Position Direction
    << static_cast<Domains::Point_vector> (that)
    // Dipole intensity
    << "I=\"" << that.weight() << "\" "
    // Cell id
    << "index_cell=\"" << that.get_cell_id_() << "\" "
    // Conductivity eigenvalues
    << "lambda1=\"" << that.lambda1() 
    << "\" lambda2=\"" << that.lambda2() 
    << "\" lambda3=\"" << that.lambda3() << "\" ";
  
  //
  //
  return stream;
}
