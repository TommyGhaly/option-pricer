#ifndef OPTION_PRICER_H
#define OPTION_PRICER_H
#include <vector>

//Binomial Tree Calculations
double binomial_tree_euro(double S, double K, double r, double q, double sigma, double T, int N, bool is_call);
double binomial_tree_american(double S, double K, double r, double q, double sigma, double T, int N, bool is_call);

//Black-Scholes Calculations
double black_scholes(double S, double X, double r, double q, double time, double sigma, bool is_call);
double d_1(double S, double X, double r, double q, double time, double sigma);
double d_2(double S, double X, double r, double q, double time, double sigma);
double N(double x);

//Greeks Calculations
double delta(double S, double K, double r, double q, double T, double sigma, bool is_call);
double gamma(double S, double K, double r, double q, double T, double sigma);
double theta(double S, double K, double r, double q, double T, double sigma, bool is_call);
double vega(double S, double K, double r, double q, double T, double sigma);
double rho(double S, double K, double r, double q, double T, double sigma, bool is_call);
double phi(double x);
double vanna(double S, double K, double r, double q, double T, double sigma);
double charm(double S, double K, double r, double q, double T, double sigma, bool is_call);
double vomma(double S, double K, double r, double q, double T, double sigma);
double veta(double S, double K, double r, double q, double T, double sigma);

//Monte Carlo Pricing
// Main pricing functions
double monte_carlo_american_option(double S, double K, double r, double q, double sigma, double T, int steps, int num_simulations, bool is_call);
double monte_carlo_asian_option(double S, double K, double r, double q, double sigma, double T, int steps, int num_simulations, bool is_call);

// Helper functions
double generate_random_normal(double mean, double sigma);
std::vector<double> generate_gpm_path(double S, double r, double q, double sigma, double dt, int steps);
double back_propagation(std::vector<std::vector<double>>& paths, double K, double r, double q, double dt, int steps, bool is_call);
std::vector<double> polynomial_regression(const std::vector<double>& X, const std::vector<double>& Y);



#endif
