#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <vector>

// Main pricing functions
double monte_carlo_american_option(double S, double K, double r, double sigma, double T, int steps, int num_simulations, bool is_call);
double monte_carlo_asian_option(double S, double K, double r, double sigma, double T, int steps, int num_simulations, bool is_call);

// Helper functions
double generate_random_normal(double mean, double sigma);
std::vector<double> generate_gpm_path(double S, double r, double sigma, double dt, int steps);
double back_propagation(std::vector<std::vector<double>>& paths, double K, double r, double dt, int steps, bool is_call);
std::vector<double> polynomial_regression(const std::vector<double>& X, const std::vector<double>& Y);

#endif
