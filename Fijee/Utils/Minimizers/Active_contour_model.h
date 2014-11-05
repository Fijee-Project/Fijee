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
#ifndef ACTIVE_CONTOUR_MODEL_H
#define ACTIVE_CONTOUR_MODEL_H
/*!
 * \file Active_contour_model.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <vector>
#include <iterator>     // std::next
#include <algorithm>    // std::sort
#include <list>
//
// UCSF
//
#include "Minimizer.h"
#include <Utils/Data_structure/Basic_point.h>
// 
// 
// 
typedef Utils::Data_structure::Basic_point<int> BPoint;
//
//
//
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \namespace Minimizers
   * 
   * Name space for our new package
   *
   */
  namespace Minimizers
  {
    /*! \class Active_contour_model
     * \brief classe Active_contour_model
     *
     *  This class is an active contour model, also called snakes, is a framework for delineating an object outline from a possibly noisy 2D image.
     *  This framework attempts to minimize an energy associated to the current contour as a sum of an internal and external energy:
     *   - The external energy is supposed to be minimal when the snake is at the object boundary position. The most straightforward approach consists in giving low values when the regularized gradient around the contour position reaches its peak value.
     *   - The internal energy is supposed to be minimal when the snake has a shape which is supposed to be relevant considering the shape of the sought object. The most straightforward approach grants high energy to elongated contours (elastic force) and to bended/high curvature contours (rigid force), considering the shape should be as regular and smooth as possible. (wiki).
     * 
     */
    class Active_contour_model : public It_minimizer
    {
    private:
      //! Points of the snake
      std::list< BPoint > snake_;
      //! Length of the snake (euclidean distance)
      double snakelength_;
      //! Size of the image X 
      int width_;
      //! Size of the image Y
      int height_;
      //! Slice Z
      int slice_z_;
      //! Gradient value
      double* gradient_;
      //! Gradient flow
      double* flow_;

      // 
      // Energy coefficients
      // coefficients for the 4 energy functions
      //! alpha = coefficient for uniformity (high => force equals distance between points)
      double alpha_;
      //! beta  = coefficient for curvature  (high => force smooth curvature)
      double beta_;
      //! gamma  = coefficient for flow      (high => force gradient attraction)
      double gamma_;
      //! delta  = coefficient for intertia  (high => get stuck to gradient)
      double delta_;



      // 
      // maximum number of iterations (if no convergence)
      int Max_iteration_;

    public:
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Active_contour_model_sphere
       *
       */
      Active_contour_model();
      /*!
       *  \brief Default Constructor
       *
       *  Constructor of the class Active_contour_model_sphere
       *
       */
      Active_contour_model( const int, const int, const int, const int*,
			    const double, bool Sobel = false );
      /*!
       *  \brief Destructeur
       *
       *  Destructor of the class Active_contour_model_sphere
       */
      virtual ~Active_contour_model();
      /*!
       *  \brief Copy Constructor
       *
       *  Constructor is a copy constructor
       *
       */
    Active_contour_model( const Active_contour_model& that):It_minimizer(that){};
      /*!
       *  \brief Operator =
       *
       *  Operator = of the class Active_contour_model_sphere
       *
       */
      Active_contour_model& operator = ( const Active_contour_model& that)
	{
	  It_minimizer::operator=(that);
	  return *this;
	};

    public:
      double get_snakelength_() 
      {
	//
	// total length of snake
	snakelength_ = 0;
	// 
	for( auto point = snake_.begin() ; point != snake_.end() ; point++ )
	  if( std::next(point,1) != snake_.end() )
	    snakelength_ += sqrt( point->squared_distance(*(std::next(point,1))) );
	
	// 
	// 
	return snakelength_;
      }

      
    public:
      /*!
       *  \brief minimize function
       *
       *  This method initialize the minimizer. 
       */
      virtual void initialization( Function,  
				   const std::vector< Estimation_tuple >&,
				   const std::vector< std::tuple<double, double> >& ){};
      /*!
       *  \brief minimize function
       *
       *  This method launch the minimization algorithm.
       *  From Pierre SchwartzProfil (http://khayyam.developpez.com/articles/algo/contours-actifs/)
       *  and http://www.developpez.net/forums/d498649/general-developpement-1/algorithme-mathematiques-2031/contribuez-416/image-snake-contour-actif/
       *
       */
      virtual void minimize();

    private:
      /*!
       *  \brief Image gradient
       *
       *  This method compute the image gradient for each vertices
       */
      void image_gradient( const int* );
      /*!
       *  \brief Sobel gradient
       *
       *  This method compute the image gradient for each vertices with Sobel filter
       */
      void Sobel_filter( const int* ){/* ToDo */};
    
      /*!
       *  \brief Image flow
       *
       *  This method compute the image flow gradient for each vertices
       */
      void image_flow( const double );
      /*!
       *  \brief Set value
       *
       *  This method check if x and y are in the array range and 
       */
      void set_value( double* Output, int X, int Y, double New_value) 
      {
	// 
	// 
	double value;
	//
	if( X >= 0 && X < width_ && Y >= 0 && Y < height_ ) 
	  {
	    value = Output[ X + Y*height_ ];
	    // 
	    if ( value > New_value )
	      Output[ X + Y*height_ ] = New_value;
	  }
      }

      // 
      // Energy functions
      // 

      // 
      // Internal energy
      // E_{int} = ( \alpha |v_{s}(s)|^{2} + \beta |v_{ss}(s)|^{2})/2
      // 
    private:
      /*!
       *  \brief 
       *
       *  This method 
       */
      double f_uniformity( const BPoint& prev, const BPoint& next, const BPoint& p ) 
      {
	//
	// length of previous segment
	double un = sqrt(prev.squared_distance( p ));
	
	// mesure of uniformity
	double avg = snakelength_ / snake_.size();
	double dun = fabs(un-avg);
	
	// elasticity energy
	return dun*dun;
      }
      /*!
       *  \brief 
       *
       *  This method 
       */
      double f_curvature( const BPoint& prev, const BPoint& p, const BPoint& next) 
      {
	// 
	// 
	int ux = p.x()-prev.x();
	int uy = p.y()-prev.y();
	double un = sqrt(ux*ux+uy*uy);
	// 
	int vx = p.x()-next.x();
	int vy = p.y()-next.y();
	double vn = sqrt(vx*vx+vy*vy);
	
	if (un == 0 || vn == 0) 
	  return 0;
	else
	  {
	    // 
	    double cx = (vx+ux)/(un*vn);
	    double cy = (vy+uy)/(un*vn);
	    // curvature energy
	    double cn = cx*cx+cy*cy;
	    return cn;
	  }
      }

      // 
      // External energy
      // E_{ext} = 
      //
      
      /*!
       *  \brief 
       *
       *  This method 
       */
     double f_gflow( const BPoint& cur, const BPoint& p) 
     {
       // 
       // gradient flow
       double 
	 dcur = flow_[cur.x() + height_*cur.y()],
	 dp   = flow_[p.x() + height_*p.y()];

       // 
       //
       return dp - dcur;
     }
     /*!
      *  \brief 
      *
      *  This method 
      */
     double f_inertia( const BPoint& cur, const BPoint& p) 
     {
       double d = sqrt( cur.squared_distance( p ) );
       double g = gradient_[cur.x() + height_*cur.y()];
       double e = g*d;

       // 
       //
       return e;
     }


    private:
      /*!
       *  \brief Iteration step
       *
       *  This method compute 
       */
      bool iteration_step();
      /*!
       *  \brief 
       *
       *  This method compute 
       */
      void remove_overlapping_points();
      /*!
       *  \brief 
       *
       *  This method compute 
       */
      void add_missing_points();
      /*!
       *  \brief 
       *
       *  This method compute 
       */
      void rebuild();
    };
    /*!
     *  \brief Dump values for Electrode
     *
     *  This method overload "<<" operator for a customuzed output.
     *
     */
    std::ostream& operator << ( std::ostream&, const Active_contour_model& );
  }
}
#endif
