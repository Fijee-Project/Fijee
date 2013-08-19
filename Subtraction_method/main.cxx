#include <vector>
#include <dolfin.h>
#include "Poisson.h"
#include "Sources.h"
#include "Conductivity.h"
#include "Boundaries.h"
#include "Sub_Domaines.h"

using namespace dolfin;

int main()
{
  //
  // Parameters
  //
  //  parameters["num_threads"] = 4;

  //
  // Create mesh and function space
  //

  //
  //
  Mesh mesh("/home/cobigo/devel/C++/UCSF/Bucket/mesh.xml");
  info(mesh);
  Poisson::FunctionSpace V(mesh);
  
  //
  // Define boundary condition
  Periphery perifery;

  //
  // Define Subdomaines
  MeshFunction< long unsigned int > domains(mesh, "/home/cobigo/devel/C++/UCSF/Bucket/mesh_subdomains.xml");
  //  CellFunction< uint > * domains = std::dynamic_cast< CellFunction< uint >* >(domains_file);

  //
  // Initialize mesh function for boundary domains
  // we tag the boundaries
  FacetFunction< size_t > boundaries(mesh);
  boundaries.set_all(0);
  perifery.mark(boundaries, 1);

  //
  // Define input data
  // 
  
  //
  // Isotrope conductivity
  Sigma_isotrope 
    a_inf(0.33);
  //
  Sigma_isotrope 
    brain_white_matter(0.33),             // # 4
    brain_gray_matter(0.33),              // # 5
    // Subcortical
    brain_brain_stem_subcortical(0.33),   // # 6
    brain_hippocampus(0.33),              // # 7
    brain_amygdala_subcortical(0.33),     // # 8
    brain_caudate(0.33),                  // # 9
    brain_putamen(0.33),                  // # 10
    brain_thalamus(0.33),                 // # 11
    brain_accumbens(0.33),                // # 12
    brain_pallidum(0.33),                 // # 13
    brain_ventricle(0.33),                // # 14
    brain_plexus(0.33),                   // # 15
    brain_fornix_subcortical(0.33),       // # 16
    brain_corpus_collosum(0.33),          // # 17
    brain_vessel(0.33),                   // # 18
    brain_cerebellum_gray_matter(0.33),   // # 19
    brain_cerebellum_white_matter(0.33),  // # 20 
    brain_ventral_diencephalon(0.33),     // # 21 
//    brain_optic_chiasm_subcortical(0.33), // # 22 
    // CSF
    csf(0./*1.79*/),                            // # 3
    // Skull and scalp
    skull(0./*0.0042*/),                        // # 2
    scalp(0./*0.33*/);                          // # 1
//  // Anisotrope conductivity
//  //  Sigma_skull a_skull;

  //
  // Source in an infinite space
  Phi Phi_0;

  //
  // Define Dirichlet boundary conditions 
  DirichletBC bc(V, Phi_0, perifery);

  //
  // Define variational forms
  Poisson::BilinearForm a(V, V);
  Poisson::LinearForm L(V);
  // Bilinear
  a.a_1   = scalp;
  a.a_2   = skull;
  a.a_3   = csf;
  a.a_4   = brain_white_matter;
  a.a_5   = brain_gray_matter;
  a.a_6   = brain_brain_stem_subcortical;
  a.a_7   = brain_hippocampus;
  a.a_8   = brain_amygdala_subcortical;
  a.a_9   = brain_caudate;
  a.a_10  = brain_putamen;
  a.a_11  = brain_thalamus;
  a.a_12  = brain_accumbens;
  a.a_13  = brain_pallidum;
  a.a_14  = brain_ventricle;
  a.a_15  = brain_plexus;
  a.a_16  = brain_fornix_subcortical;
  a.a_17  = brain_corpus_collosum;
  a.a_18  = brain_vessel;
  a.a_19  = brain_cerebellum_gray_matter;
  a.a_20  = brain_cerebellum_white_matter;
  a.a_21  = brain_ventral_diencephalon;
//  a.a_22  = ;
  //
  a.dx   = domains;
  // Linear
  L.a_inf = a_inf;
  //
  L.a_1  = scalp;
  L.a_2  = skull;
  L.a_3  = csf;
  L.a_4  = brain_white_matter;
  L.a_5  = brain_gray_matter;
  L.a_6  = brain_brain_stem_subcortical;
  L.a_7  = brain_hippocampus;
  L.a_8  = brain_amygdala_subcortical;
  L.a_9  = brain_caudate;
  L.a_10  = brain_putamen;
  L.a_11  = brain_thalamus;
  L.a_12  = brain_accumbens;
  L.a_13  = brain_pallidum;
  L.a_14  = brain_ventricle;
  L.a_15  = brain_plexus;
  L.a_16  = brain_fornix_subcortical;
  L.a_17  = brain_corpus_collosum;
  L.a_18  = brain_vessel;
  L.a_19  = brain_cerebellum_gray_matter;
  L.a_20  = brain_cerebellum_white_matter;
  L.a_21  = brain_ventral_diencephalon;
  //
  L.Phi_0 = Phi_0;
  L.dx    = domains;
  L.ds    = boundaries;

  //
  // Compute solution
  Function u(V);
  // solve(a == L, u, bc);
  solve(a == L, u);
  
  //
  // View the solution
  //
  // Save solution in VTK format
  File file("poisson.pvd");

//  //
//  // Theoric potential in an infinit medium
//  Function Phi0_th(V);
//  Phi0_th.interpolate(Phi_0);
//  plot(Phi0_th, "Theoritical Potential");
//  file << Phi0_th;
//
//  Function Potential_total(V);
//  *Potential_total.vector()  = *u.vector();
//  *Potential_total.vector() += *Phi0_th.vector();
//  //  *diff.vector() /= *phi0_th.vector();
//  file << Potential_total;
  file << u;
  file << domains;
////  // Plot solution
////  plot(u, "Phi_s");
////  plot(Potential_total, "Total Potential");
////  plot(boundaries);
//  plot(domains);
//  //  plot(mesh);
//  interactive();

  //
  //
  return 0;
}
