#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

double black_scholes(double S, double X, double r, double q, double time, double sigma, bool is_call);
double d_1(double S, double X, double r, double q, double time, double sigma);
double d_2(double S, double X, double r, double q, double time, double sigma);
double N(double x);


#endif
