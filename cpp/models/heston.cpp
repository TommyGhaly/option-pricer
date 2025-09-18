#include "../include/models.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <complex>
#include <functional>

/**
 * Returns u_j coefficients
 * u_1 = 0.5 (for P1)
 * u_2 = -0.5 (for P2)
 */
double u(int num) {
    if (num == 1) {
        return 0.5;
    } else {
        return -0.5;
    }
}

/**
 * Computes b_j coefficients
 * b_1 = κ + λ - ρσ (for P1)
 * b_2 = κ + λ (for P2)
 */
double b(int num, double lambda, double kappa,
         double rho, double sigma) {
    if (num == 1) {
        return kappa + lambda - (sigma * rho);
    } else {
        return kappa + lambda;
    }
}

/**
 * Computes d_j = sqrt[(ρσφi - b_j)² - σ²(2u_jφi - φ²)]
 * This is the discriminant in the Heston characteristic function
 */
std::complex<double> d(int num, double lambda, double kappa,
                       double rho, double sigma, double phi) {
    std::complex<double> i(0.0, 1.0);

    double b_val = b(num, lambda, kappa, rho, sigma);
    double u_val = u(num);

    // Calculate the expression under the square root
    std::complex<double> term1 = rho * sigma * phi * i - b_val;
    std::complex<double> discriminant = term1 * term1 -
                                       sigma * sigma * (2.0 * u_val * phi * i - phi * phi);

    // Take square root with correct branch
    std::complex<double> result = std::sqrt(discriminant);

    // Ensure we choose the branch with positive real part for stability
    if (result.real() < 0) {
        result = -result;
    }

    return result;
}

/**
 * Computes g_j = (b_j - ρσφi + d_j) / (b_j - ρσφi - d_j)
 * This ratio appears in the Heston characteristic function
 */
std::complex<double> g(int num, double lambda, double kappa,
                       double rho, double sigma, double phi) {
    std::complex<double> i(0.0, 1.0);

    double b_val = b(num, lambda, kappa, rho, sigma);
    std::complex<double> d_val = d(num, lambda, kappa, rho, sigma, phi);

    // Calculate numerator and denominator
    std::complex<double> base = b_val - rho * sigma * phi * i;
    std::complex<double> numerator = base + d_val;
    std::complex<double> denominator = base - d_val;

    // Check for potential division by zero
    if (std::abs(denominator) < 1e-12) {
        // Handle near-zero denominator case
        // When denominator ≈ 0, g ≈ -1 (from L'Hopital's rule)
        return std::complex<double>(-1.0, 0.0);
    }

    return numerator / denominator;
}

/**
 * Computes the D_j(T,φ) function in the Heston characteristic function
 * D_j represents the coefficient of V_0 in the characteristic function exponent
 */
std::complex<double> d_function(int num, double lambda, double kappa,
                                double rho, double sigma, double phi,
                                double T) {
    std::complex<double> i(0.0, 1.0);

    // Get components
    double b_val = b(num, lambda, kappa, rho, sigma);
    std::complex<double> d_val = d(num, lambda, kappa, rho, sigma, phi);
    std::complex<double> g_val = g(num, lambda, kappa, rho, sigma, phi);

    // Calculate numerator and denominator for stability
    std::complex<double> numerator = b_val - rho * sigma * phi * i + d_val;
    std::complex<double> exp_dT = std::exp(d_val * T);
    std::complex<double> denominator = 1.0 - g_val * exp_dT;

    // Check for numerical stability
    if (std::abs(denominator) < 1e-10) {
        // Use L'Hopital's rule or Taylor expansion for small denominator
        // When g*exp(d*T) ≈ 1, use series expansion
        return numerator / (sigma * sigma) * T * exp_dT / (1.0 + g_val * exp_dT);
    }

    // Standard formula: D_j = (b_j - ρσφi + d_j)/(σ²) * (1 - e^(d_j*T))/(1 - g_j*e^(d_j*T))
    return numerator / (sigma * sigma) * (1.0 - exp_dT) / denominator;
}

/**
 * Computes the C_j(T,φ) function in the Heston characteristic function
 * C_j represents the constant term in the characteristic function exponent
 */
std::complex<double> c_function(int num, double lambda, double kappa,
                                double rho, double sigma, double phi,
                                double T, double theta, double r,
                                double q) {
    std::complex<double> i(0.0, 1.0);

    double b_val = b(num, lambda, kappa, rho, sigma);
    std::complex<double> d_val = d(num, lambda, kappa, rho, sigma, phi);
    std::complex<double> g_val = g(num, lambda, kappa, rho, sigma, phi);

    // First term: (r-q)φiT
    std::complex<double> term1 = (r - q) * phi * i * T;

    // Calculate the logarithm term carefully
    std::complex<double> exp_dT = std::exp(d_val * T);
    std::complex<double> numerator_log = 1.0 - g_val * exp_dT;
    std::complex<double> denominator_log = 1.0 - g_val;

    // Handle potential numerical issues with the logarithm
    std::complex<double> log_term;
    if (std::abs(denominator_log) < 1e-10 || std::abs(numerator_log) < 1e-10) {
        // Use Taylor expansion for log((1-g*e^(dT))/(1-g)) when g ≈ 1
        std::complex<double> x = g_val * (exp_dT - 1.0);
        log_term = x - x*x/2.0 + x*x*x/3.0; // Taylor series
    } else {
        log_term = std::log(numerator_log / denominator_log);
    }

    // Second term: (κθ/σ²)[(b_j - ρσφi + d_j)T - 2ln((1-g_j*e^(d_j*T))/(1-g_j))]
    std::complex<double> term2 = (kappa * theta) / (sigma * sigma) *
                                 ((b_val - rho * sigma * phi * i + d_val) * T - 2.0 * log_term);

    return term1 + term2;
}

/**
 * Computes the Heston characteristic function f_j(φ)
 * f_j(φ) = exp(C_j + D_j*V_0 + iφ*ln(S))
 */
std::complex<double> f(int num, double lambda, double kappa,
                       double rho, double sigma, double theta,
                       double r, double q, double S,
                       double phi, double T, double v0) {
    std::complex<double> i(0.0, 1.0);

    // Get C and D functions
    std::complex<double> C = c_function(num, lambda, kappa, rho, sigma,
                                        phi, T, theta, r, q);
    std::complex<double> D = d_function(num, lambda, kappa, rho, sigma, phi, T);

    // Return exp(C + D*V_0 + iφ*ln(S))
    std::complex<double> exponent = C + D * v0 + i * phi * std::log(S);

    // Check for overflow/underflow
    if (exponent.real() > 700.0) {
        // Prevent overflow
        return std::complex<double>(std::exp(700.0), 0.0);
    } else if (exponent.real() < -700.0) {
        // Prevent underflow
        return std::complex<double>(0.0, 0.0);
    }

    return std::exp(exponent);
}

/**
 * Computes the risk-neutral probability P1 or P2 for the Heston model
 * using numerical integration (Simpson's rule)
 *
 * @param num: 1 for P1, 2 for P2
 * @param N: Number of integration points (higher = more accurate)
 * @param L: Not used in this implementation (was for COS method truncation)
 * @return: Probability value in [0,1]
 */
double p(int num, double lambda, double kappa, double rho,
         double sigma, double theta, double r,
         double q, double S, double T,
         double v0, double K, int N, double L) {
    std::complex<double> i(0.0, 1.0);

    // This implements the integral: P_j = 0.5 + (1/π) ∫ Re[e^(-iφln(K)) * f_j(φ) / (iφ)] dφ

    double integral = 0.0;
    double phi_min = 1e-8;  // Avoid singularity at φ = 0
    double phi_max = 100.0; // Upper limit for integration

    // Simpson's rule setup
    double h = (phi_max - phi_min) / (N - 1);

    // Ensure N is odd for Simpson's rule
    if (N % 2 == 0) N++;

    for (int k = 0; k < N; ++k) {
        double phi = phi_min + k * h;

        // Get characteristic function f_j(φ)
        std::complex<double> char_func = f(num, lambda, kappa, rho, sigma,
                                          theta, r, q, S, phi, T, v0);

        // Compute integrand: e^(-iφln(K)) * f(φ) / (iφ)
        std::complex<double> numerator = std::exp(-i * phi * std::log(K)) * char_func;
        std::complex<double> denominator = i * phi;
        std::complex<double> integrand_val = numerator / denominator;

        // Simpson's rule weights: 1, 4, 2, 4, 2, ..., 4, 1
        double weight;
        if (k == 0 || k == N-1) {
            weight = h / 3.0;
        } else if (k % 2 == 1) {
            weight = 4.0 * h / 3.0;
        } else {
            weight = 2.0 * h / 3.0;
        }

        integral += weight * integrand_val.real();
    }

    // Final probability: P_j = 0.5 + integral/π
    double probability = 0.5 + integral / M_PI;

    // Ensure probability is in valid range [0,1]
    return std::max(0.0, std::min(1.0, probability));
}

/**
 * Main Heston model option pricing function
 * Uses the risk-neutral valuation formula:
 * Call = S*exp(-qT)*P1 - K*exp(-rT)*P2
 * Put = Call - S*exp(-qT) + K*exp(-rT) (put-call parity)
 *
 * @param S: Current stock price
 * @param K: Strike price
 * @param r: Risk-free rate
 * @param q: Dividend yield
 * @param T: Time to maturity
 * @param v0: Initial variance
 * @param kappa: Mean reversion speed
 * @param theta: Long-term variance mean
 * @param sigma: Volatility of variance
 * @param rho: Correlation between stock and variance
 * @param lambda: Market price of volatility risk (usually 0)
 * @param is_call: true for call option, false for put option
 */
double heston_model(double S, double K, double r,
                    double q, double T, double v0,
                    double kappa, double theta, double sigma,
                    double rho, double lambda, bool is_call) {
    // Calculate probabilities P1 and P2
    double P1 = p(1, lambda, kappa, rho, sigma, theta, r, q, S, T, v0, K, 512, 10.0);
    double P2 = p(2, lambda, kappa, rho, sigma, theta, r, q, S, T, v0, K, 512, 10.0);

    // Calculate call option price: C = S*e^(-qT)*P1 - K*e^(-rT)*P2
    double call_price = S * std::exp(-q * T) * P1 - K * std::exp(-r * T) * P2;

    // Ensure non-negative call price
    call_price = std::max(0.0, call_price);

    // Return call price or convert to put using put-call parity
    if (is_call) {
        return call_price;
    } else {
        // Put-call parity: P = C - S*e^(-qT) + K*e^(-rT)
        double put_price = call_price - S * std::exp(-q * T) + K * std::exp(-r * T);
        return std::max(0.0, put_price);
    }
}
