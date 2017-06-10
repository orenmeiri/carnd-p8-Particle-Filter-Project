/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;

	double std_x = std[0];
	double std_y = std[1];
	double std_psi = std[2];

	num_particles = 100;

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_psi);

	for (int i = 0; i < num_particles; i++) {
		double sample_x, sample_y, sample_theta;

		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle new_particle;

		new_particle.id = i;
		new_particle.x = sample_x;
		new_particle.y = sample_y;
		new_particle.theta = sample_theta;

		particles.push_back(new_particle);
	}

  is_initialized= true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    if (fabs(yaw_rate) == 0) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double weight_temp = 1 / (2 * M_PI * std_x * std_y);

  for (int i = 0; i < num_particles; i++) {
    Particle particle = particles[i];
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    double particle_weight = 1;
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs observation = observations[j];
      double obs_landmark_x, obs_landmark_y;
      // apply observation to particle location
      obs_landmark_x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
      obs_landmark_y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;

      // find nearest map landmark to observation
      double best_distance = std::numeric_limits<double>::max();
      int best_landmark_id = -1;
      for (int l = 0; l < map_landmarks.landmark_list.size(); l++) {
        Map::single_landmark_s landmark = map_landmarks.landmark_list[l];
        double distance = dist(obs_landmark_x, obs_landmark_y, landmark.x_f, landmark.y_f);
        if (distance > sensor_range){
          continue;
        }
        if (distance < best_distance) {
          best_distance = distance;
          best_landmark_id = l;
        }
      }
      double best_landmark_x = map_landmarks.landmark_list[best_landmark_id].x_f;
      double best_landmark_y = map_landmarks.landmark_list[best_landmark_id].y_f;

      // calculate observation error contribution to particle weight for next resample
      double w = weight_temp * exp(-(
                                           (pow(obs_landmark_x - best_landmark_x, 2)/(2 * std_x * std_x)) +
                                           (pow(obs_landmark_y - best_landmark_y, 2)/(2 * std_y * std_y))
                                    ));
      particle_weight = particle_weight * w;

      associations.push_back(best_landmark_id);
      sense_x.push_back(obs_landmark_x);
      sense_y.push_back(obs_landmark_y);
    }

    // apply weight to particle based on accumulated 'error' from observations
    particles[i].weight = particle_weight;

    SetAssociations(particle, associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> new_particles;

  // resampling wheel
  double w_max = 0;
  for (int i = 0; i < num_particles; i++) {
    if (particles[i].weight > w_max) {
      w_max = particles[i].weight;
    }
  }

  int index = rand() % num_particles;

  double b = 0;
  for (int i = 0; i < num_particles; i++) {
    b = b + rand() * 2 * w_max / RAND_MAX;
    while (particles[index].weight < b) {
      b = b - particles[index].weight;
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
