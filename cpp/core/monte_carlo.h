#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

double monte_carlo_american_option(double S, double K, double r, double sigma, double T, int steps, int num_simulations, bool is_call);
double monte_carlo_asian_option(double S, double K, double r, double sigma, double T, int steps, int num_simulations, bool is_call);

#endif
