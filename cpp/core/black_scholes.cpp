#include "black_scholes.h"
#include <cmath>
#include <iostream>

double d_1(double S, double X, double r, double q, double time, double sigma) {
    // continuous dividend yield q included via forward price adjustment
    return ((log(S / X)) + (r - q + (0.5 * sigma * sigma)) * time) / (sigma * sqrt(time));
}

double d_2(double S, double X, double r, double q, double time, double sigma) {
    return ((log(S / X)) + (r - q - (0.5 * sigma * sigma)) * time) / (sigma * sqrt(time));
}

double N(double x) {
    return std::erfc(-x / sqrt(2)) / 2;
}

double black_scholes(double S, double X, double r, double q, double time, double sigma, bool is_call) {
    double d1 = d_1(S, X, r, q, time, sigma);
    double d2 = d_2(S, X, r, q, time, sigma);
    // adjust spot by continuous dividend yield via e^{-qT}
    if (is_call) {
        return (S * std::exp(-q * time) * N(d1)) - (X * exp(-r * time) * N(d2));
    } else {
        return (X * exp(-r * time) * N(-d2)) - (S * std::exp(-q * time) * N(-d1));
    }
}

extern "C" {
    double bs_option_price(double S, double X, double r, double q,
                           double time, double sigma, int is_call) {
        // convert int to bool
        return black_scholes(S, X, r, q, time, sigma, is_call != 0);
    }
}
