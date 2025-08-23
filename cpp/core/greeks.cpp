#include "greeks.h"
#include "black_scholes.h"
#include <iostream>
#include <cmath>

double phi(double x) {
    return (1.0 / sqrt( 2 * M_PI)) * std::exp(-0.5 * x * x);
}

double delta(double S, double K, double r, double q, double T, double sigma, bool is_call) {
    double ND_1 = N(d_1(S, K, r, q, T, sigma));
    if (is_call) {
        return std::exp(-q * T) * ND_1; // adjust for continuous dividend yield
    } else {
        return std::exp(-q * T) * (ND_1 - 1);
    }
}

double gamma(double S, double K, double r, double q, double T, double sigma) {
    return (phi(d_1(S, K, r, q, T, sigma))) / (S * sigma * sqrt(T)) * std::exp(-q * T);
}

double theta(double S, double K, double r, double q, double T, double sigma, bool is_call) {
    double d1 = d_1(S, K, r, q, T, sigma);
    double d2 = d_2(S, K, r, q, T, sigma);
    double term1 = -(S * std::exp(-q * T) * phi(d1) * sigma) / (2 * sqrt(T));
    double term2_call = q * S * std::exp(-q * T) * N(d1);
    double term2_put = q * S * std::exp(-q * T) * N(-d1);
    double term3_call = r * K * std::exp(-r * T) * N(d2);
    double term3_put = r * K * std::exp(-r * T) * N(-d2);

    if (is_call) {
        return term1 - term2_call - term3_call;
    } else {
        return term1 + term2_put + term3_put;
    }
}

double vega(double S, double K, double r, double q, double T, double sigma) {
    return S * std::exp(-q * T) * phi(d_1(S, K, r, q, T, sigma)) * sqrt(T);
}


double rho(double S, double K, double r, double q, double T, double sigma, bool is_call) {
    double term = K * T * std::exp(-r * T);
    double d2 = d_2(S, K, r, q, T, sigma);
    if (is_call) {
        return term * N(d2);
    } else {
        return -term * N(-d2);
    }
}
double vanna(double S, double K, double r, double q, double T, double sigma){
    double d1 = d_1(S, K, r, q, T, sigma);
    double d2 = d_2(S, K, r, q, T, sigma);
    return -phi(d1) * std::exp(-q * T) * d2 / sigma;
}
double charm(double S, double K, double r, double q, double T, double sigma, bool is_call){
    double d1 = d_1(S, K, r, q, T, sigma);
    double d2 = d_2(S, K, r, q, T, sigma);
    double sqrt_T = std::sqrt(T);

    double common_term = phi(d1) * ((r - q) / (sigma * sqrt_T) - d2 / (2 * T));

    if (is_call) {
        return -std::exp(-q * T) * (q * N(d1) + common_term);
    } else {
        return -std::exp(-q * T) * (q * N(-d1) - common_term);
    }
}
double vomma(double S, double K, double r, double q, double T, double sigma){
    double d1 = d_1(S, K, r, q, T, sigma);
    double d2 = d_2(S, K, r, q, T, sigma);
    return S * std::exp(-q * T) * phi(d1) * sqrt(T) * d1 * d2 / sigma;
}
double veta(double S, double K, double r, double q, double T, double sigma){
    double d1 = d_1(S, K, r, q, T, sigma);
    double d2 = d_2(S, K, r, q, T, sigma);
    return -S * std::exp(-q * T) * phi(d1) * std::sqrt(T) * (q + ((r - q) * d1) / (sigma * std::sqrt(T)) - (1 + d1 * d2) / (2 * T));
}
