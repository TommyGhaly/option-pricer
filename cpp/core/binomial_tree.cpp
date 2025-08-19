#include "binomial_tree.h"
#include <iostream>
#include <cmath>
#include <vector>

double priceOptionBinTree(double S, double K, double r, double sigma, double T, int N, bool is_call) {
    double dt = T / N; // Time step
    double u = std::exp(sigma * sqrt(dt)); // Up factor
    double d = 1.0 / u; // Down factor
    double p = (std::exp(r * dt) -d) / (u -d); // Risk-neutral probability
    double discount = std::exp(-r * dt); // Discount factor

    // initialize finishing prices at maturity
    std::vector<double> assetsPrices( N + 1);

    for (int i = 0; i <= N; ++i) {
        assetsPrices[i] = S * std::pow(u, N - i) * std::pow(d, i);
    }


    // initialize option values at maturity
    std::vector<double> optionValues(N + 1);
    for (int i = 0; i <= N; ++i) {
        if (is_call) {
            optionValues[i] = std::max(0.0, assetsPrices[i] -K);
        } else {
            optionValues[i] = std::max(0.0, K - assetsPrices[i]);
        }
    }

    // Backward induction to calculate option price at time 0
    for (int step = N - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            optionValues[i] = (p * optionValues[i] + (1 - p) * optionValues[i + 1]) * discount;
        }
    }
    return optionValues[0]; // Return the option price at time 0
}
