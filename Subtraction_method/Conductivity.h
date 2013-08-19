#ifndef _CONDUCTIVITY_H
#define _CONDUCTIVITY_H
#include <dolfin.h>
#include <vector>

using namespace dolfin;

//
// Conductivity values
//

// Isotrope conductivity
class Sigma_isotrope : public Expression
{
 private:
  double sigma_;

 public:
  // Create expression with 3 components
 Sigma_isotrope( double Sigma = 1) : Expression(3,3), sigma_(Sigma) {}

  // 
  void eval(Array<double>& values, const Array<double>& x) const
  {
    //
    values[0] = sigma_; values[3] = 0.0;    values[6] = 0.0;
    values[1] = 0.0;    values[4] = sigma_; values[7] = 0.0;
    values[2] = 0.0;    values[5] = 0.0;    values[8] = sigma_;
  }
  
  // Rank
  virtual std::size_t value_rank() const
  {
    return 2;
  }

  // Dimension
  virtual std::size_t value_dimension(uint i) const
  {
    return 3;
  }
};

// Skull
class Sigma_skull : public Expression
{
 public:
  // Create expression with 3 components
 Sigma_skull() : Expression(2,2) {}

  void eval(Array<double>& values, const Array<double>& x) const
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
  
  // Rank
  virtual std::size_t value_rank() const
    {
      return 2;
    }
  
  // Dimension
  virtual std::size_t value_dimension(uint i) const
    {
      return 2;
    }
};
#endif
