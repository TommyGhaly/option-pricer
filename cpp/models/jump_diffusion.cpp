#include "../include/models.h"
#include "../core/black_scholes.h"
#include <iostream>
#include <cmath>

// Fixed factorial function
int factorial(int n) {
    if (n <= 1) return 1;
    int total = 1;
    for(int i = 2; i <= n; ++i) {
        total *= i;
    }
    return total;
}

// For large n, use log-space to avoid overflow
double P(int n, double lambda, double T) {
    if (n == 0) return std::exp(-lambda * T);

    // For n > 20, factorial might overflow, so use log-space
    if (n > 20) {
        double log_prob = -lambda * T + n * std::log(lambda * T);
        for (int i = 2; i <= n; ++i) {
            log_prob -= std::log(i);
        }
        return std::exp(log_prob);
    }

    // For small n, direct calculation is fine
    return (std::exp(-lambda * T) * std::pow(lambda * T, n)) / factorial(n);
}

// Fixed parentheses
double sigma_n(int n, double sigma, double sigma_j, double T) {
    return std::sqrt(sigma * sigma + (n * sigma_j * sigma_j) / T);
}

// This is correct since q is handled in black_scholes
double r_n(int n, double r, double lambda_prime, double mu_j, double T) {
    return r - lambda_prime + (n * mu_j) / T;
}

int determine_num_n(double lambda, double T) {
    double expected_jumps = lambda * T;
    int estimate_n = (int)(expected_jumps + 4 * std::sqrt(expected_jumps) + 5);
    return std::max(20, std::min(estimate_n, 200));
}

double jump_diffusion(double S, double K, double r, double q, double sigma,
                     double lambda, double mu_j, double sigma_j, double T, bool is_call) {

    double k = std::exp(mu_j + 0.5 * sigma_j * sigma_j) - 1;
    double lambda_prime = lambda * k;

    double total_price = 0.0;
    int max_n = determine_num_n(lambda, T);

    for(int n = 0; n < max_n; ++n) {
        double prob_n = P(n, lambda, T);
        double vol_n = sigma_n(n, sigma, sigma_j, T);
        double rate_n = r_n(n, r, lambda_prime, mu_j, T);

        // black_scholes handles dividend yield internally
        double bs_price = black_scholes(S, K, rate_n, q, T, vol_n, is_call);

        total_price += prob_n * bs_price;

        // Optional: early termination for efficiency
        if (prob_n * bs_price < 1e-10 && n > 10) {
            break;
        }
    }

    return total_price;
}
