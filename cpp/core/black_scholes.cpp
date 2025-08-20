#include "black_scholes.h"
#include <cmath>
#include <iostream>

double d_1(double S, double X, double r, double time, double sigma) {
    return ((log(S / X)) + (r + (0.5 * sigma * sigma)) * time) / (sigma * sqrt(time));
}

double d_2(double S, double X, double r, double time, double sigma) {
    return ((log(S / X)) + (r - (0.5 * sigma * sigma)) * time) / (sigma * sqrt(time));
}

double N(double x) {
    return std::erfc(-x / sqrt(2)) / 2;
}

double black_scholes(double S, double X, double r, double time, double sigma, bool is_call) {
    double d1 = d_1(S, X, r, time, sigma);
    double d2 = d_2(S, X, r, time, sigma);
    if (is_call) {
        return (S * N(d1)) - (X * exp(-r * time) * N(d2));
    } else {
        return (X * exp(-r * time) * N(-d2)) - (S * N(-d1));
    }
}
