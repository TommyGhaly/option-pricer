#include "../include/models.h"
#include "../include/option_pricer.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

// Numerical derivatives for Dupire formula
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
    double price_minus = black_scholes(S, K - dk, r, q, T, sigma, is_call);
    double price = black_scholes(S, K, r, q, T, sigma, is_call);
    return (price_plus - 2 * price + price_minus) / (dk * dk);
}

// Calculate local volatility at a single point
double local_volatility_point(double S, double K, double r, double q, double sigma, double T, bool is_call) {
    double dt = 0.001;
    double dk = 0.01 * K;  // 1% of strike

    double dC_dT_val = dC_dT(S, K, r, q, sigma, T, is_call, dt);
    double dC_dK_val = dC_dK(S, K, r, q, sigma, T, is_call, dk);
    double d2C_dK2_val = d2C_dK2(S, K, r, q, sigma, T, is_call, dk);

    double numerator = dC_dT_val + r * K * dC_dK_val;
    double denominator = 0.5 * K * K * d2C_dK2_val;

    // Avoid division by zero or negative values
    if (denominator <= 1e-10) {
        return sigma;  // Return implied vol as fallback
    }

    double local_var = numerator / denominator;
    if (local_var <= 0) {
        return sigma;  // Return implied vol as fallback
    }

    return std::sqrt(local_var);
}

// Thomas algorithm for solving tridiagonal systems
std::vector<double> solve_tridiagonal(const std::vector<double>& a,
                                    const std::vector<double>& b,
                                    const std::vector<double>& c,
                                    const std::vector<double>& d) {
    int n = d.size();
    std::vector<double> c_star(n-1);
    std::vector<double> d_star(n);
    std::vector<double> x(n);

    // Forward sweep
    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n-1; i++) {
        c_star[i] = c[i] / (b[i] - a[i-1] * c_star[i-1]);
    }

    for (int i = 1; i < n; i++) {
        d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / (b[i] - a[i-1] * c_star[i-1]);
    }

    // Back substitution
    x[n-1] = d_star[n-1];
    for (int i = n-2; i >= 0; i--) {
        x[i] = d_star[i] - c_star[i] * x[i+1];
    }

    return x;
}

// Linear interpolation
double interpolate(const std::vector<double>& x_values,
                  const std::vector<double>& y_values,
                  double x) {
    auto it = std::lower_bound(x_values.begin(), x_values.end(), x);

    if (it == x_values.end()) {
        return y_values.back();
    }
    if (it == x_values.begin()) {
        return y_values.front();
    }

    int idx = it - x_values.begin();
    double x1 = x_values[idx-1];
    double x2 = x_values[idx];
    double y1 = y_values[idx-1];
    double y2 = y_values[idx];

    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
}

// Base FDM solver for local volatility model
double local_vol_fdm_base(double S0, double K, double r, double q, double T,
                         double implied_vol, bool is_call, bool is_american,
                         int N_S = 200, int N_T = 100) {

    // Grid parameters
    double S_max = 3.0 * K;
    double S_min = 0.0;
    double dS = (S_max - S_min) / N_S;
    double dt = T / N_T;

    // Create grids
    std::vector<double> S(N_S + 1);
    std::vector<std::vector<double>> V(N_S + 1, std::vector<double>(N_T + 1));

    // Initialize stock price grid
    for (int i = 0; i <= N_S; i++) {
        S[i] = S_min + i * dS;
    }

    // Terminal condition
    for (int i = 0; i <= N_S; i++) {
        if (is_call) {
            V[i][N_T] = std::max(S[i] - K, 0.0);
        } else {
            V[i][N_T] = std::max(K - S[i], 0.0);
        }
    }

    // Backward time stepping
    for (int j = N_T - 1; j >= 0; j--) {
        double t = j * dt;

        // Boundary conditions
        if (is_call) {
            V[0][j] = 0.0;  // S = 0
            V[N_S][j] = S_max - K * std::exp(-r * (T - t));  // S = S_max
        } else {
            V[0][j] = K * std::exp(-r * (T - t));  // S = 0 for put
            V[N_S][j] = 0.0;  // S = S_max for put
        }

        // Setup tridiagonal system for interior points
        int n = N_S - 1;
        std::vector<double> a(n-1), b(n), c(n-1), d(n);

        for (int i = 1; i < N_S; i++) {
            // For now, use implied vol directly
            double sigma = implied_vol;
            double sigma2 = sigma * sigma;

            // Finite difference coefficients (implicit scheme)
            double alpha = 0.5 * dt * ((r - q) * S[i] / dS - sigma2 * S[i] * S[i] / (dS * dS));
            double beta = 1.0 + dt * (sigma2 * S[i] * S[i] / (dS * dS) + r);
            double gamma = -0.5 * dt * ((r - q) * S[i] / dS + sigma2 * S[i] * S[i] / (dS * dS));

            // Fill matrix coefficients
            if (i > 1) {
                a[i-2] = -alpha;
            }
            b[i-1] = beta;
            if (i < N_S - 1) {
                c[i-1] = -gamma;
            }

            // RHS
            d[i-1] = V[i][j+1];
        }

        // Adjust for boundary conditions
        if (n > 0) {
            double sigma_1 = implied_vol;
            double sigma2_1 = sigma_1 * sigma_1;
            double alpha_1 = 0.5 * dt * ((r - q) * S[1] / dS - sigma2_1 * S[1] * S[1] / (dS * dS));
            d[0] += alpha_1 * V[0][j];

            if (n > 1) {
                double sigma_n = implied_vol;
                double sigma2_n = sigma_n * sigma_n;
                double gamma_n = -0.5 * dt * ((r - q) * S[N_S-1] / dS + sigma2_n * S[N_S-1] * S[N_S-1] / (dS * dS));
                d[n-1] -= gamma_n * V[N_S][j];
            }
        }

        // Solve tridiagonal system
        std::vector<double> solution = solve_tridiagonal(a, b, c, d);

        // Copy solution back
        for (int i = 1; i < N_S; i++) {
            V[i][j] = solution[i-1];

            // American option early exercise check
            if (is_american) {
                double intrinsic = is_call ? std::max(S[i] - K, 0.0) : std::max(K - S[i], 0.0);
                V[i][j] = std::max(V[i][j], intrinsic);
            }
        }
    }

    // Extract price at S0
    std::vector<double> V_now(N_S + 1);
    for (int i = 0; i <= N_S; i++) {
        V_now[i] = V[i][0];
    }

    return interpolate(S, V_now, S0);
}

// European option with local volatility
double european_local_vol_fdm(double S0, double K, double r, double q, double T,
                             double implied_vol, bool is_call,
                             int N_S = 200, int N_T = 100) {
    return local_vol_fdm_base(S0, K, r, q, T, implied_vol, is_call, false, N_S, N_T);
}

// American option with local volatility
double american_local_vol_fdm(double S0, double K, double r, double q, double T,
                             double implied_vol, bool is_call,
                             int N_S = 200, int N_T = 100) {
    return local_vol_fdm_base(S0, K, r, q, T, implied_vol, is_call, true, N_S, N_T);
}
