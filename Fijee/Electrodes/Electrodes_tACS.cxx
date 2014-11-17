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
#include <cmath> 
#include <ctgmath>
//
// UCSF
//
#include "Electrodes_tACS.h"
#define PI 3.14159265359
// 
// 
// 
Electrodes::Electrodes_tACS::Electrodes_tACS():
  Fijee::XML_writer("electrodes_tACS.xml"),
  I_tot_plus_(0.), I_tot_minus_(0.), time_step_(0.001 /*s*/), 
  time_start_(0.), time_end_(0.), ramp_up_(0.), ramp_down_(0.)
{}
// 
// 
// 
Electrodes::Electrodes_tACS::Electrodes_tACS( const std::vector< std::tuple< std::string/*label*/, double/*frequency*/, double/*injection*/, double/*oscillation*/, double/*phase*/ > > Electrodes_plus,
					      const std::vector< std::tuple< std::string/*label*/, double/*frequency*/, double/*injection*/, double/*oscillation*/, double/*phase*/ > > Electrodes_minus,
					      const double Elapse_time, const double Starting_time,
					      const double Ramp_up, const double Ramp_down ):
  Fijee::XML_writer("electrodes_tACS.xml"),
  I_tot_plus_(0.), I_tot_minus_(0.), time_step_(0.001 /*s*/), 
  time_start_(Starting_time), time_end_(Starting_time+Elapse_time), 
  ramp_up_(Ramp_up), ramp_down_(Ramp_down)
{
  // 
  // 
  int number_of_steps = static_cast<int>( Elapse_time/time_step_ );
  // 
  std::vector< double/* intensity */ > I_tot_plus_ts(number_of_steps, 0.);
  std::map< std::string, double > return_electrodes_fraction;

  // 
  // Ramping process
  // 

  // 
  // 
  if( Elapse_time < ramp_up_ + ramp_down_ )
    {
      std::cerr << "The ramping process over range the elapse time! " << std::endl;
      std::cerr << "Ramp_up + Ramp_down MUST be less or equal to the elapse time." 
		<< std::endl;
      //
      abort();
    }


  // 
  // Check on electrode injections
  // 

  // 
  // 
  for ( auto electrode :  Electrodes_plus )
    {
      if ( std::get<2>(electrode) >= 0. )
	{
	  if( electrodes_.find( std::get<0>(electrode) ) == electrodes_.end() )
	    {
	      Electrode elec;
	      elec.label_     = std::get<0>(electrode);
	      elec.nu_        = std::get<1>(electrode);
	      elec.I_         = std::get<2>(electrode);
	      elec.amplitude_ = std::get<3>(electrode);
	      elec.phase_     = std::get<4>(electrode);
	      // 
	      electrodes_.insert( std::make_pair(elec.label_, elec) );
	      I_tot_plus_ += elec.I_;
	      // 
	      intensity_time_series_[ elec.label_ ] = 
		std::vector< std::tuple<double,double> >(number_of_steps);
	    }
	  else
	    {
	      std::cerr << "Electrode " << std::get<0>(electrode) 
			<< " has already been recorded."
			<< std::endl;
	      //
	      abort();
	    }    
	}
      else
	{
	  std::cerr << "All positive electrodes must have a positive injection" << std::endl;
	  std::cerr << "I_{+} = " << std::get<2>(electrode)
		    << std::endl;
	  //
	  abort();
	}
    }    
  // 
  for ( auto electrode :  Electrodes_minus )
    {
      if ( std::get<2>(electrode) <= 0. )
	{
	  if( electrodes_.find( std::get<0>(electrode) ) == electrodes_.end() )
	    {
	      Electrode elec;
	      elec.label_     = std::get<0>(electrode);
	      elec.nu_        = std::get<1>(electrode);
	      elec.I_         = std::get<2>(electrode);
	      elec.amplitude_ = std::get<3>(electrode);
	      elec.phase_     = std::get<4>(electrode);
	      // 
	      electrodes_.insert( std::make_pair(elec.label_, elec) );
	      I_tot_minus_ += elec.I_;
	      // frctionnal contribution of the electrode
	      return_electrodes_fraction[elec.label_] = elec.I_ / I_tot_plus_;
	      // 
	      intensity_time_series_[ elec.label_ ] = 
		std::vector< std::tuple<double,double> >(number_of_steps);
	    }
	  else
	    {
	      std::cerr << "Electrode " << std::get<0>(electrode) 
			<< " has already been recorded."
			<< std::endl;
	      //
	      abort();
	    }    
	}
      else
	{
	  std::cerr << "All negative electrodes must have a negative injection" << std::endl;
	  std::cerr << "I_{-} = " << std::get<2>(electrode)
		    << std::endl;
	  //
	  abort();
	}
    }
  // 
  if( I_tot_plus_ != - I_tot_minus_ )
    {
      std::cerr << "The Sum of I_{+} and (-I_{-}) injections is not null" << std::endl;
      std::cerr << "I_{+} = " << I_tot_plus_
		<< " and I_{-} = " << I_tot_minus_ << std::endl;
      //
      abort();
    }

  // 
  // Production of the AC time series injection
  // 
  
  // 
  // Positive electrodes contribution
  for( int time = 0 ; time < number_of_steps ; time++ )
    for( auto& electrode : intensity_time_series_ )
      if( electrodes_[electrode.first].I_ >= 0.  )
	{
	  // 
	  // 
	  double time_s = time  * time_step_;

	  // 
	  // Injection value at the time step
	  double injection_value =  electrodes_[electrode.first].amplitude_;
	  injection_value *=  sin( 2 * PI * electrodes_[electrode.first].nu_ * time_s + 
				   electrodes_[electrode.first].phase_ );
	  injection_value += electrodes_[electrode.first].I_;
	  injection_value *= ramping_process(time_start_ + time_s);
	  
	  // 
	  // Records
	  (electrode.second)[time] = 
	    std::make_tuple(time_start_+ time_s, injection_value);

	  // 
	  // Total injection at each time step
	  I_tot_plus_ts[time] += injection_value;
      }
  
  // 
  // Negative electrodes contribution
  for( int time = 0 ; time < number_of_steps ; time++ )
    for( auto& electrode : intensity_time_series_ )
      if( electrodes_[electrode.first].I_ < 0.  )
	{
	  // 
	  // 
	  double time_s = time  * time_step_;

	  // 
	  // Injection value at the time step
	  double injection_value =  I_tot_plus_ts[time] * return_electrodes_fraction[electrode.first];
	  // s -> ms
	  (electrode.second)[time] = 
	    std::make_tuple(time_start_+ time_s, injection_value);
      }
}
// 
// 
// 
void
Electrodes::Electrodes_tACS::Make_analysis()
{
#ifdef FIJEE_TRACE
#if FIJEE_TRACE == 100
  // 
  // 
  output_stream_.precision(10);
  std::size_t number_of_steps = intensity_time_series_.begin()->second.size();

  //
  // R header
  // 
  output_stream_
    << "time ";
  // 
  for ( auto electrode : intensity_time_series_)
    output_stream_ <<  electrode.first << " ";
  // 
  output_stream_ << std::endl;

  // 
  // R values
  // 
  for( int step = 0 ; step < number_of_steps ; step++ )
    {  
      output_stream_
	<< std::get<0>( (intensity_time_series_.begin()->second)[step] ) << " ";
      for( auto time_step : intensity_time_series_ )
	output_stream_
	  << std::get<1>( (time_step.second)[step] ) << " ";
      // 
      output_stream_ << std::endl;      
    }

  //
  //
  Make_output_file("electrodes_tACS.frame");
#endif
#endif      
}
// 
// 
// 
void 
Electrodes::Electrodes_tACS::output_XML( const std::string files_path_output )
{
  // 
  // Build XML output 
  // 
  
  // 
  // 
  int number_electrodes = 0;


  //
  // Read the electrodes xml file
  std::cout << "Load electrodes file" << std::endl;
  //
  std::string electrodes_xml = files_path_output + std::string("electrodes.xml");
  //
  pugi::xml_document     xml_file;
  pugi::xml_parse_result result = xml_file.load_file( electrodes_xml.c_str() );
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
	std::map< std::string/*label*/, Electrode >::iterator it_electrode;
	// loop over the samples
	for ( auto sample : setup_node )
	  {
	    //
	    // Get the number of electrodes
	    number_electrodes = sample.attribute("size").as_int();

	    //
	    //
	    for( auto electrode : sample )
	      {
		// 
		// 
		std::string label = electrode.attribute("label").as_string(); 
		
		// 
		// 
		if( (it_electrode = electrodes_.find(label)) != electrodes_.end() )
		  {
		    it_electrode->second.index_ = electrode.attribute("index").as_uint();
		    // position
		    it_electrode->second.position_x_ = electrode.attribute("x").as_double(); /* m */
		    it_electrode->second.position_y_ = electrode.attribute("y").as_double(); /* m */
		    it_electrode->second.position_z_ = electrode.attribute("z").as_double(); /* m */
		    // Direction
		    it_electrode->second.direction_vx_ = electrode.attribute("vx").as_double();
		    it_electrode->second.direction_vy_ = electrode.attribute("vy").as_double();
		    it_electrode->second.direction_vz_ = electrode.attribute("vz").as_double();
		    // Potential
		    double I = electrode.attribute("I").as_double(); /* Ampere */
		    if ( I != 0. )
		      std::cerr << "WARNING: electrodes.xml file has been compromized!\n" 
				<< "Electrode: " << label << " has I != 0." 
				<< std::endl;
		    it_electrode->second.V_ = electrode.attribute("V").as_double(); /* Volt */
		    // Impedance
		    it_electrode->second.Re_z_l_ = electrode.attribute("Re_z_l").as_double();
		    it_electrode->second.Im_z_l_ = electrode.attribute("Im_z_l").as_double();
		    // Contact surface between Electrode and the scalp
		    it_electrode->second.type_    = electrode.attribute("type").as_string(); 
		    it_electrode->second.surface_ = electrode.attribute("surface").as_double(); /* m^2 */
		    it_electrode->second.radius_  = electrode.attribute("radius").as_double();  /* m */
		    it_electrode->second.height_  = electrode.attribute("height").as_double();  /* m */
		  }
		else
		  {
		    Electrode elec;
		    // 
		    elec.index_ = electrode.attribute("index").as_uint();
		    // position
		    elec.position_x_ = electrode.attribute("x").as_double(); /* m */
		    elec.position_y_ = electrode.attribute("y").as_double(); /* m */
		    elec.position_z_ = electrode.attribute("z").as_double(); /* m */
		    // Direction
		    elec.direction_vx_ = electrode.attribute("vx").as_double();
		    elec.direction_vy_ = electrode.attribute("vy").as_double();
		    elec.direction_vz_ = electrode.attribute("vz").as_double();
		    // Potential
		    elec.label_ = label;
		    elec.I_ = electrode.attribute("I").as_double(); /* Ampere */
		    if ( elec.I_ != 0. )
		      std::cerr << "WARNING: electrodes.xml file has been compromized!" << std::endl;
		    elec.V_ = electrode.attribute("V").as_double(); /* Volt */
		    // Impedance
		    elec.Re_z_l_ = electrode.attribute("Re_z_l").as_double();
		    elec.Im_z_l_ = electrode.attribute("Im_z_l").as_double();
		    // Contact surface between Electrode and the scalp
		    elec.type_    = electrode.attribute("type").as_string(); 
		    elec.surface_ = electrode.attribute("surface").as_double(); /* m^2 */
		    elec.radius_  = electrode.attribute("radius").as_double();  /* m */
		    elec.height_  = electrode.attribute("height").as_double();  /* m */
		    // 
		    electrodes_.insert( std::make_pair(label, elec) );
		  }
	      }
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
  // Output generation
  // 

  // 
  // XML output
  set_file_name_( files_path_output + "electrodes_tACS.xml" );

  // 
  // 
  std::size_t number_of_steps = intensity_time_series_.begin()->second.size();
  // 
  auto setup_node = fijee_.append_child("setup");
  setup_node.append_attribute("size") = static_cast<int>( number_of_steps );
  // 
  for( int step = 0 ; step < number_of_steps ; step++ )
    {
      auto electrodes_node = setup_node.append_child("electrodes");
      electrodes_node.append_attribute("index") = step;
      electrodes_node.append_attribute("time")  = 
	std::get<0>((intensity_time_series_.begin()->second)[step]); 
      electrodes_node.append_attribute("size") = number_electrodes;
      // 
      for(auto electrode : electrodes_)
	{
	  auto electrode_node = electrodes_node.append_child("electrode");
	  // 
	  electrode_node.append_attribute("index")   = electrode.second.index_;
	  electrode_node.append_attribute("x")       = electrode.second.position_x_;
	  electrode_node.append_attribute("y")       = electrode.second.position_y_;
	  electrode_node.append_attribute("z")       = electrode.second.position_z_;
	  electrode_node.append_attribute("vx")      = electrode.second.direction_vx_;
	  electrode_node.append_attribute("vy")      = electrode.second.direction_vy_;
	  electrode_node.append_attribute("vz")      = electrode.second.direction_vz_;
	  electrode_node.append_attribute("label")   = electrode.second.label_.c_str();
	  // 
	  auto it_injection = intensity_time_series_.find(electrode.second.label_);
	  if( it_injection != intensity_time_series_.end() )
	    electrode_node.append_attribute("I")     = std::get<1>((it_injection->second)[step]);
	  else
	    electrode_node.append_attribute("I")     = 0.;
	  electrode_node.append_attribute("V")       = electrode.second.V_;
	  electrode_node.append_attribute("Re_z_l")  = electrode.second.Re_z_l_;;
	  electrode_node.append_attribute("Im_z_l")  = electrode.second.Im_z_l_;
	  electrode_node.append_attribute("type")    = electrode.second.type_.c_str();
	  electrode_node.append_attribute("radius")  = electrode.second.radius_;
	  electrode_node.append_attribute("height")  = electrode.second.height_;
	  electrode_node.append_attribute("surface") = electrode.second.surface_;;
	}
    }


  // 
  // Statistical analysise
  Make_analysis();
}
// 
// 
// 
double 
Electrodes::Electrodes_tACS::ramping_process( const double Time )
{
  if( Time >= time_start_ && Time < time_start_ + ramp_up_ )
    return ( Time - time_start_ ) / ramp_up_;
  else if( Time >= time_start_ + ramp_up_ && Time <= time_end_ - ramp_down_ )
    return 1.;
  else if( Time > time_end_ - ramp_down_ && Time <= time_end_ )
    return ( time_end_ - Time ) / ramp_down_;
  else
    return 0.;
}; 
