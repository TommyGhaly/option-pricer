#ifndef GREEKS_H
#define GREEKS_H

double delta(double S, double K, double r, double q, double T, double sigma, bool is_call);
double gamma(double S, double K, double r, double q, double T, double sigma);
double theta(double S, double K, double r, double q, double T, double sigma, bool is_call);
double vega(double S, double K, double r, double q, double T, double sigma);
double rho(double S, double K, double r, double q, double T, double sigma, bool is_call);
double phi(double x);

#endif
