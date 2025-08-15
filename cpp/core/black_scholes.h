#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

double calculate_option_price(double S, double X, double r, double time, double sigma, bool is_call);

#endif