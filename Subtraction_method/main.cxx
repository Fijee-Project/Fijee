#include <vector>
#include <dolfin.h>
#include "Poisson.h"
#include "Sources.h"
#include "Boundaries.h"
#include "Sub_Domaines.h"

using namespace dolfin;

int main()
{
  //
  // Create mesh and function space
  //
  //  UnitCircleMesh mesh(64);
  Mesh mesh("/home/cobigo/Softwares/Fenics/share/dolfin/data/meshes/mesh.xml");
  info(mesh);
  Poisson::FunctionSpace V(mesh);
  
  //
  // Define boundary condition
  //
  Periphery perifery;

  //
  // Define Subdomaines
  //
  Brain brain;
  CSF   csf;
  Skull skull;
  Scalp scalp;
  //
  // Initialize mesh function for interior domains
  // We tag the cells
  //
  CellFunction< size_t > domains(mesh);
  // All domaine tags 0
  domains.set_all(0);
  // Obstacles tags 1
  brain.mark(domains, 1);
  csf.mark(domains, 2);
  skull.mark(domains, 3);
  scalp.mark(domains, 4);

  //  //
  //  for ( CellIterator cell(mesh) ; !cell.end() ; ++cell ){
  //    Point p = (*cell).midpoint();
  //    //    if ( obstacle.inside( p.coordinates(), true) )
  //    if( domains[*cell] == 0  )
  //      domains[*cell] = 4;
  //  }


  //
  // Initialize mesh function for boundary domains
  // we tag the boundaries
  //
  FacetFunction< size_t > boundaries(mesh);
  boundaries.set_all(0);
  perifery.mark(boundaries, 1);

  //
  // Define input data
  // 
  Constant     a_1(0.33);
  Constant     a_2(1.79);
  Constant     a_3(0.0042);
  Constant     a_4(0.33);
  Constant     a_inf = a_1;
  Phi          Phi_0;

  //
  // Define Dirichlet boundary conditions at top and bottom boundaries
  //
  DirichletBC bc(V, Phi_0, perifery);

  //
  // Define variational forms
  //
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);

  // Bilinear
  a.a_1  = a_1;
  a.a_2  = a_2;
  a.a_3  = a_3;
  a.a_4  = a_4;
  a.dx  = domains;

  // Linear
  L.a_inf  = a_inf;
  L.a_1    = a_1;
  L.a_2    = a_2;
  L.a_3    = a_3;
  L.a_4    = a_4;
  L.Phi_0 = Phi_0;
  L.dx    = domains;
  L.ds    = boundaries;

  //
  // Compute solution
  //
  Function u(V);
  // solve(a == L, u, bc);
  solve(a == L, u);
  
  // Save solution in VTK format
  File file("poisson.pvd");

  // Theoric Potential
  Function Phi0_th(V);
  Phi0_th.interpolate(Phi_0);
  plot(Phi0_th, "Theoritical Potential");

  Function Potential_total(V);
  *Potential_total.vector()  = *u.vector();
  *Potential_total.vector() += *Phi0_th.vector();
  //  *diff.vector() /= *phi0_th.vector();
  file << Potential_total;
  file << domains;
  // Plot solution
  plot(u, "Phi_s");
  plot(Potential_total, "Total Potential");
  //  plot(boundaries);
  //  plot(domains);
  //  plot(mesh);
  interactive();

  return 0;
}
