#include "binomial_tree.h"
#include <iostream>
#include <cmath>
#include <vector>

double binomial_tree_euro(double S, double K, double r, double q, double sigma, double T, int N, bool is_call) {
    double dt = T / N;
    double u = std::exp(sigma * sqrt(dt));
    double d = 1.0 / u;
    // risk-neutral probability adjusted for continuous dividend yield q
    double p = (std::exp((r - q) * dt) - d) / (u - d);
    double discount = std::exp(-r * dt);

    // Cache powers of u and d
    std::vector<double> u_powers(N + 1);
    std::vector<double> d_powers(N + 1);
    u_powers[0] = 1.0;
    d_powers[0] = 1.0;
    for (int i = 1; i <= N; ++i) {
        u_powers[i] = u_powers[i-1] * u;
        d_powers[i] = d_powers[i-1] * d;
    }

    // Single vector for option values (memory efficient)
    std::vector<double> optionValues(N + 1);

    // Initialize option values at maturity
    for (int i = 0; i <= N; ++i) {
    double S_final = S * u_powers[N-i] * d_powers[i];
        if (is_call) {
            optionValues[i] = std::max(0.0, S_final - K);
        } else {
            optionValues[i] = std::max(0.0, K - S_final);
        }
    }

    // Backward induction - update in place
    for (int step = N - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            optionValues[i] = (p * optionValues[i] + (1 - p) * optionValues[i + 1]) * discount;
        }
    }

    return optionValues[0];
}

double binomial_tree_american(double S, double K, double r, double q, double sigma, double T, int N, bool is_call) {
    double dt = T / N;
    double u = std::exp(sigma * sqrt(dt));
    double d = 1.0 / u;
    double p = (std::exp((r - q) * dt) - d) / (u - d);
    double discount = std::exp(-r * dt);

    // Cache powers of u and d
    std::vector<double> u_powers(N + 1);
    std::vector<double> d_powers(N + 1);
    u_powers[0] = 1.0;
    d_powers[0] = 1.0;
    for (int i = 1; i <= N; ++i) {
        u_powers[i] = u_powers[i-1] * u;
        d_powers[i] = d_powers[i-1] * d;
    }

    // Single vector for option values (memory efficient)
    std::vector<double> optionValues(N + 1);

    // Initialize option values at maturity
    for (int i = 0; i <= N; ++i) {
        double S_final = S * u_powers[N-i] * d_powers[i];
        if (is_call) {
            optionValues[i] = std::max(0.0, S_final - K);
        } else {
            optionValues[i] = std::max(0.0, K - S_final);
        }
    }

    // Backward induction with early exercise check
    for (int step = N - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            // Calculate asset price at this node using cached powers
            double S_current = S * u_powers[step-i] * d_powers[i];

            // Intrinsic value
            double intrinsic = is_call ? std::max(0.0, S_current - K)
                                       : std::max(0.0, K - S_current);

            // Continuation value
            double continuation = (p * optionValues[i] + (1 - p) * optionValues[i + 1]) * discount;

            // Max of exercise vs hold
            optionValues[i] = std::max(intrinsic, continuation);
        }
    }

    return optionValues[0];
}

extern "C" {
    double bt_euro(double S, double K, double r, double q, double sigma,
                   double T, int N, int is_call) {
        return binomial_tree_euro(S, K, r, q, sigma, T, N, is_call != 0);
    }

    double bt_american(double S, double K, double r, double q, double sigma,
                       double T, int N, int is_call) {
        return binomial_tree_american(S, K, r, q, sigma, T, N, is_call != 0);
    }
}
