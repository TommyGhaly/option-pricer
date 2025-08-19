#include "greeks.h"
#include "black_scholes.h"
#include <iostream>
#include <cmath>

double phi(double x) {
    return (1.0 / sqrt( 2 * M_PI)) * std::exp(-0.5 * x * x);
}

double delta(double S, double K, double r, double T, double sigma, bool is_call) {
    double ND_1 = N(d_1(S, K, r, T, sigma));
    if (is_call) {
        return ND_1;
    } else {
        return ND_1 - 1; // For put options, delta is negative of call delta
    }
}

double gamma(double S, double K, double r, double T, double sigma) {
    return (phi(d_1(S, K, r, T, sigma))) / (S * sigma * sqrt(T));
}

double theta(double S, double K, double r, double T, double sigma, bool is_call) {
    double term1 = -(S * phi(d_1(S, K, r, T, sigma)) * sigma) / (2 * sqrt(T));
    double term2 = r * K * exp(-r * T) * N(d_2(S, K, r, T, sigma));
    if (is_call) {
        return term1 - term2;
    } else {
        return term1 + term2; // For put options, theta is adjusted by the second term
    }
}

double vega(double S, double K, double r, double T, double sigma) {
    return S * phi(d_1(S, K, r, T, sigma)) * sqrt(T);
}


double rho(double S, double K, double r, double T, double sigma, bool is_call) {
    double term = K * T * exp(-r * T);
    double d2 = d_2(S, K, r, T, sigma);
    if (is_call) {
        return term * N(d2);
    } else {
        return -term * N(-d2); // For put options, rho is negative of call rho
    }
}
