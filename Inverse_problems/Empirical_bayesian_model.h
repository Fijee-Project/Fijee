//  Copyright (c) 2014, Yann Cobigo 
//  All rights reserved.     
//   
//  Redistribution and use in source and binary forms, with or without       
//  modification, are permitted provided that the following conditions are met:   
//   
//  1. Redistributions of source code must retain the above copyright notice, this   
//     list of conditions and the following disclaimer.    
//  2. Redistributions in binary form must reproduce the above copyright notice,   
//     this list of conditions and the following disclaimer in the documentation   
//     and/or other materials provided with the distribution.   
//   
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;   
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND   
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   
//     
//  The views and conclusions contained in the software and documentation are those   
//  of the authors and should not be interpreted as representing official policies,    
//  either expressed or implied, of the FreeBSD Project.  
#ifndef _EMPIRICAL_BAYESIAN_MODEL_H
#define _EMPIRICAL_BAYESIAN_MODEL_H
#include <list>
#include <tuple>
#include <string>
#include <stdexcept>      // std::logic_error
// 
// Eigen
//
#include <Eigen/Dense>
typedef Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > MatrixXd;
typedef Eigen::Matrix< double, Eigen::Dynamic, 1 > VectorXd;
//
// pugixml
// same resources than Dolfin
//
#include "Utils/pugi/pugixml.hpp"
//
// UCSF project
//
#include "Utils/Fijee_environment.h"
#include "Utils/Fijee_log_management.h"
#include "Inverse_solver.h"
//
//
//
/*!
 * \file Empirical_bayesian_model.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Solver
 * 
 * Name space for our new package
 *
 */
namespace Inverse
{
  /*! \class Empirical_bayesian_model
   * \brief classe representing the source localisation with subtraction method.
   *
   *  This class representing the Physical model for the source localisation using the subtraction method.
   */
  template< typename Noise >
    class Empirical_bayesian_model : Inverse_solver
    {
    private:
      //! Potential measure
      MatrixXd potential_;
      //! Leadfield matrix
      MatrixXd leadfield_;
      //! Dipole actictivation random vector estimation
      MatrixXd dipole_activation_;
      //! Number of dipoles
      int number_dipoles_;
      //! Noise construction object
      Noise noise_;

      // 
      // Empirical model
      //
      //! Prior covariance matrix on the random variable J (dipole activation vector)
      //      Eigen::DiagonalMatrix<double, Eigen::Dynamic> Gamma_;
      MatrixXd Gamma_;
      //! Marginal likelihood covariance
      MatrixXd likelihood_covariance_;
      //! Empirical covariance
      MatrixXd empirical_covariance_;
   
 
    public:
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Empirical_bayesian_model
       *
       */
      Empirical_bayesian_model();
      /*!
       *  \brief Copy Constructor
       *
       *  Constructor is a copy constructor
       *
       */
      Empirical_bayesian_model( const Empirical_bayesian_model& ){};
      /*!
       *  \brief Destructeur
       *
       *  Destructor of the class Empirical_bayesian_model
       */
      virtual ~Empirical_bayesian_model(){/* Do nothing */};
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class Empirical_bayesian_model
       *
       */
      Empirical_bayesian_model& operator = ( const Empirical_bayesian_model& ){return *this;};
      /*!
       *  \brief Operator ()
       *
       *  Operator () of the class Empirical_bayesian_model
       *
       */
      virtual void operator () (){};
    
    public:
      /*!
       *  \brief Get number of physical events
       *
       *  This method return the number of parallel process for the Physics solver. 
       *  In the case of source localization the number of events is the number of dipoles simulated.
       *
       */
      virtual inline
	int get_number_of_physical_events(){return number_dipoles_;};
      /*!
       *  \brief Dipole activity estimation
       *
       *  This method implement the Champagne algorithm.
       *  
       *
       */
      void dipole_activity_estimation();
    };
  //
  // 
  // 
  template< typename Noise >
    Empirical_bayesian_model< Noise >::Empirical_bayesian_model():Inverse_solver()
    {
      // 
      // Time log
      FIJEE_TIME_PROFILER("Empirical_bayesian_model constructor");

      //
      // 
      int 
	Nj = 0, // number of dipoles
	Ne = 0, // number of probes (electrodes)
	Nt = 0; // number of samples
      // temporary leadfield matrix read from XML
      std::list< std::tuple< int, VectorXd > > leadfild_list;


      // 
      // Read the leadfild matrix
      // 
      std::cout << "Load leadfild matrix" << std::endl;
      //
      std::string leadfield_xml = (IIp::get_instance())->get_files_path_measure_();
      leadfield_xml += "eeg_forward.xml";

      //
      pugi::xml_document xml_file;
      pugi::xml_parse_result result = xml_file.load_file( leadfield_xml.c_str() );
      //
      switch( result.status )
	{
	case pugi::status_ok:
	  {
	    //
	    // Check that we have a FIJEE XML file
	    const pugi::xml_node fijee_node = xml_file.child("fijee");
	    if (!fijee_node)
	      {
		std::cerr << "Read data from XML: Not a FIJEE XML file" << std::endl;
		exit(1);
	      }

	    // 
	    // Get sampling
	    const pugi::xml_node setup_node = fijee_node.child("setup");
	    if (!setup_node)
	      {
		std::cerr << "Read data from XML: no setup node" << std::endl;
		exit(1);
	      }

	    // Get the number of samples
	    // and loop over the samples
	    Nt = setup_node.attribute("size").as_uint();
	    int dipole_index = 0;
	    for ( auto sample : setup_node )
	      {
		//
		// 
		Ne           = sample.attribute("size").as_uint();
		dipole_index = sample.attribute("dipole").as_uint();

		//
		//
		VectorXd potential_dipole;
		for( auto electrode : sample )
		  {
		    potential_dipole.resize(Ne);
		    int index = electrode.attribute("index").as_uint();
//		    // Label
//		    std::string label = electrode.attribute("label").as_string(); 
//		    // Intensity
//		    double I = electrode.attribute("I").as_double(); /* Ampere */
		    // Potential
		    double V = electrode.attribute("V").as_double(); /* Volt */
		    //
		    potential_dipole(index) = V;
		  }

		// 
		// record the potential from the dipole
		leadfild_list.push_back( std::make_tuple (dipole_index, potential_dipole) );
	      }

	    //
	    //
	    break;
	  };
	default:
	  {
	    std::cerr << "Error reading XML file: " << result.description() << std::endl;
	    exit(1);
	  }
	}

      // 
      // Build the leadfield matrix
      leadfield_.resize(Ne, Nj = leadfild_list.size() );
      // 
      for( auto dipole : leadfild_list )
	leadfield_.col( std::get<0>(dipole) ) = std::get<1>(dipole);

      // 
      // Initialization
      // 
      dipole_activation_.resize(Nj, Nt);
      Gamma_.setIdentity(Nj,Nj);
      //
      potential_.resize(Ne, Nt);

      // 
      // Marginal likelihood covariance matrix initialization
      likelihood_covariance_.resize(Ne, Ne);
//      // 
//      likelihood_covariance_  = noise_.get_covariance();
//      likelihood_covariance_ += leadfield_ * leadfield_.transpose();
      // 
      empirical_covariance_.resize(Ne, Ne);
      empirical_covariance_ = potential_ * potential_.transpose();

      // 
      // debug output
//#ifdef DEBUG
#ifdef TRACE
#if ( TRACE == 1 )
      std::cout << Ne << " " << Nt << " " << Nj << std::endl;
      std::cout << leadfield_ << std::endl;
      std::cout << Gamma_ << " " << std::endl;
#endif
#endif
//#endif
    }
  //
  // 
  // 
  template< typename Noise > 
    void
    Empirical_bayesian_model< Noise >::dipole_activity_estimation()
    {
      // 
      // Time log
      FIJEE_TIME_PROFILER("Empirical_bayesian_model::dipole_activity_estimation()");

      // 
      // [Nj x Ne]
      Eigen::Matrix< 
	double, 
	leadfield_.cols(), 
	leadfield_.rows() >  leadfield_T;
      //
      leadfield_T = leadfield_.transpose();
      // [Ne x Ne]
      Eigen::Matrix< 
	double, 
	likelihood_covariance_.rows(), 
	likelihood_covariance_.cols() >  likelihood_covariance_Inv;
      //
      likelihood_covariance_Inv = likelihood_covariance_.inverse();

      // 
      // Cost function
      double 
	L = 10.,
	L_old = 1.;
      // tolerance
      double epsilon = 5./* % */;
      double ratio   = 100./* % */;
      
      //
      // 
      while( ratio > epsilon && L > L_old )
	{
	  //
	  L_old = L;

	  // 
	  // X: [Nj x 1]
	  Eigen::Matrix< double, dipole_activation_.rows(), 1 > X;
	  X = Gamma_ * leadfield_T * likelihood_covariance_Inv * potential_;
	  // Z: [Nj x Nj]
	  Eigen::Matrix< double, dipole_activation_.rows(), dipole_activation_.rows() > Z;
	  // 
	  Z = leadfield_T * likelihood_covariance_Inv * leadfield_;
	  
	  // 
	  // Compute the new prior
	  for (int i = 0 ; i < Gamma_.rows() ; i++ )
	    Gamma_(i,i) = X(i,0) / sqrt( Z(i,i) );
	  
	  // 
	  // Cost function update
	  // likelihood covariance update
	  likelihood_covariance_  = noise_.get_covariance();
	  likelihood_covariance_ += leadfield_ * Gamma_ * leadfield_T;
	  //
	  likelihood_covariance_Inv = likelihood_covariance_.inverse();
	  
	  // 
	  // cost function update
	  L  = log( likelihood_covariance_.determinant() );
	  L += (empirical_covariance_ * likelihood_covariance_Inv).trace();
	  // ration update
	  ratio = abs(L - L_old) * 100. / L_old; 
	}

      // 
      // Estimation
      dipole_activation_  = Gamma_ * leadfield_T;
      dipole_activation_ *= (noise_.get_covariance() + leadfield_ * Gamma_ * leadfield_T).inverse();
      dipole_activation_ *= potential_;
    }
}
#endif
