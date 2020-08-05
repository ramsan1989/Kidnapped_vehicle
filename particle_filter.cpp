/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  num_particles = 100;  // TODO: Set the number of particles
  particles = std::vector<Particle>(static_cast<unsigned long>(num_particles));
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int m = 0; m < num_particles; ++m) {
    particles[m].x = dist_x(gen);
    particles[m].y = dist_y(gen);
    particles[m].theta = dist_theta(gen);
    particles[m].id = m;
    particles[m].weight = 1.0;
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
 
  for(int m = 0; m < num_particles;++m){
    if(fabs(yaw_rate<0.0001))
    {
      particles[m].x += velocity * delta_t * cos(particles[m].theta);
	  particles[m].y += velocity * delta_t * sin(particles[m].theta);
    }
    else{
    	particles[m].x = particles[m].x + (velocity/yaw_rate)*(sin(particles[m].theta+yaw_rate*delta_t)-sin(particles[m].theta));
    	particles[m].y = particles[m].y + (velocity/yaw_rate)*(cos(particles[m].theta)-cos(particles[m].theta+yaw_rate*delta_t));
    	particles[m].theta = particles[m].theta + yaw_rate*delta_t;
    }
    normal_distribution<double> dist_x(0, std_pos[0]);
  	normal_distribution<double> dist_y(0, std_pos[1]);
  	normal_distribution<double> dist_theta(0, std_pos[2]);
    particles[m].x = particles[m].x+ dist_x(gen);
    particles[m].y = particles[m].y + dist_y(gen);
    particles[m].theta = particles[m].theta + dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   
   */
  
  for(unsigned int i = 0; i<observations.size();++i){
    LandmarkObs current_obs = observations[i];
    double min_dist = std::numeric_limits<double>::max();
    int map_id = -1;
    for(unsigned int j = 0; j<predicted.size();++j){
      LandmarkObs current_pred = predicted[j];
      double current_dist = dist(current_obs.x, current_obs.y, current_pred.x, current_pred.y);
       if (current_dist < min_dist) {
        min_dist = current_dist;
        map_id = current_pred.id;
      }
      
    }
    observations[i].id = map_id;
    
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
double sigma_x_2 = std_landmark[0]*std_landmark[0];
double sigma_y_2 = std_landmark[1]*std_landmark[1];
double mult_factor = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  for (int i =0; i<num_particles;++i){
    double part_x = particles[i].x;
    double part_y = particles[i].y;
    double part_theta = particles[i].theta;
    
    vector<LandmarkObs> predictions;
    for (unsigned int j =0; j<map_landmarks.landmark_list.size();++j){
      double dist_x= map_landmarks.landmark_list[j].x_f - part_x;
      double dist_y = map_landmarks.landmark_list[j].y_f - part_y;
      if((dist_x*dist_x+dist_y*dist_y)<=sensor_range*sensor_range){
        predictions.push_back(LandmarkObs{ map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f });
      }
    }
    
    vector<LandmarkObs> tobs;
    for (unsigned int k =0; k<observations.size();++k){
      double tobs_x = part_x + (cos(part_theta)*observations[k].x) - (sin(part_theta)*observations[k].y);
	  double tobs_y = part_y+ (sin(part_theta)*observations[k].x) + (cos(part_theta)*observations[k].y);
	  tobs.push_back(LandmarkObs{ observations[k].id, tobs_x, tobs_y });
      
    }
    
    dataAssociation(predictions,tobs);
    vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
    particles[i].weight = 1.0;
    for (unsigned int k = 0;k<tobs.size();++k){
      int associated_id = tobs[k].id;
      double x_m_mu_x,y_m_mu_y;
      for (unsigned int j = 0; j<predictions.size();++j){
      	if(predictions[j].id==associated_id){
          x_m_mu_x = tobs[k].x - predictions[j].x;
          y_m_mu_y = tobs[k].y - predictions[j].y;
        }
      }
        double temp_weight = mult_factor * exp( -( x_m_mu_x*x_m_mu_x/(2*sigma_x_2) + (y_m_mu_y*y_m_mu_y/(2*sigma_y_2)) ) );
        if (temp_weight==0){
          temp_weight = 0.00001;
          particles[i].weight = particles[i].weight*temp_weight;
        }
        else{
        particles[i].weight = particles[i].weight*temp_weight;
          }
      		associations.push_back(associated_id);
			sense_x.push_back(tobs[k].x);
			sense_y.push_back(tobs[k].y);
        }
    
   SetAssociations(particles[i], associations, sense_x, sense_y);
  }
  

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Update weights`
	weights.clear();
	for(int i=0; i<num_particles; ++i){
		weights.push_back(particles[i].weight);
	}
    std::default_random_engine gen;
	discrete_distribution<int> particle_dist(weights.begin(),weights.end());

	// Resample particles
	vector<Particle> new_particles;
	new_particles.resize(num_particles);
	for(int i=0; i<num_particles; ++i){

		auto index = particle_dist(gen);
		new_particles[i] = std::move(particles[index]);
	}
	particles = std::move(new_particles);

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}