#include "../../Point_vector.h"

int
main()
{
  //
  //
  std::cout << "Algebra" << std::endl;
  Domains::Point_vector V1(1.,2.,3.,10.,0.,0.);
  Domains::Point_vector V2(1.,2.,3.,5.,2.,0.);
  //
  std::cout << V1.get_norm_() << std::endl;
  std::cout << V2.get_norm_() << std::endl;
  std::cout << V1.dot(V2) << std::endl;
  std::cout << V1.cosine_theta(V2) << std::endl;
  std::cout << V1.cross(V2).vx() << " "  
	    << V1.cross(V2).vy() << " "   
	    << V1.cross(V2).vz() 
	    << std::endl;

  //
  //
  std::cout << "Somme" << std::endl;
  Domains::Point_vector P1(0.,2.,4.,1.,0.,0.);
  Domains::Point_vector P2(7.,3.,4.,0.,2.,0.);
  Domains::Point_vector P3(7.,8.,3.,0.,0.,3.);
  Domains::Point_vector P4(1.,5.,9.,4.,0.,0.);
  //
  std::cout << (P1).get_norm_() << std::endl;
  std::cout << (P1 + P2).vx() << " " << (P1 + P2).vy() << " " << (P1 + P2).vz()
	    << std::endl;
  std::cout << (P1 += P2).get_norm_() << std::endl;
  std::cout << (P1).get_norm_()
	    << std::endl;
  std::cout << (P3 - P4).vx() << " " << (P3 - P4).vy() << " " << (P3 - P4).vz()
	    << std::endl;





  //
  //
  return EXIT_SUCCESS;
}
