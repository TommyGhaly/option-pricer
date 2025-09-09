#include "../include/models.h"
#include "../include/option_pricer.h"
#include <iostream>
#include <cmath>

double dC_dT(double S, double K, double r, double q, double sigma, double T, bool is_call, double dt) {
    double price_plus = black_scholes(S, K, r, q, T + dt, sigma, is_call);
    double price_minus = black_scholes(S, K, r, q, T - dt, sigma, is_call);
    return (price_plus - price_minus) / (2 * dt);
}

double dC_dK(double S, double K, double r, double q, double sigma, double T, bool is_call, double dk) {
    double price_plus = black_scholes(S, K + dk, r, q, T, sigma, is_call);
    double price_minus = black_scholes(S, K - dk, r, q, T, sigma, is_call);
    return (price_plus - price_minus) / (2 * dk);
}

double d2C_dK2(double S, double K, double r, double q, double sigma, double T, bool is_call, double dk) {
    double price_plus = black_scholes(S, K + dk, r, q, T, sigma, is_call);
    double price_minus = black_scholes(S, K -dk, r, q, T, sigma, is_call);
    double price = black_scholes(S, K, r, q, T, sigma, is_call);
    return (price_plus - (2 * price) + price_minus) / (dk * dk);
}

double local_volatility(double S, double K, double r, double q, double sigma, double T, bool is_call){
    double local_sigma = std::sqrt(dC_dT(S, K, r, q, sigma, T, is_call, 0.001) + r * K * dC_dK(S, K, r, q, sigma, T, is_call, 0.001) / 0.5 * K * K * d2C_dK2(S, K, r, q, sigma, T, is_call, 0.0001));

}
