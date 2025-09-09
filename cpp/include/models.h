#ifndef MODELS_H
#define MODELS_H


double heston_model(double S, double K, double r, double q, double T, double v0, double kappa, double theta, double sigma, double rho, double lambda, bool is_call);
double jump_diffusion(double S, double K, double r, double q, double sigma, double lambda, double mu_j, double sigma_j, double T, bool is_call);
double local_volatility(double S, double K, double r, double q, double sigma, double T, bool is_call);
#endif // MODELS_H
