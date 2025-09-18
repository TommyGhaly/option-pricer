#include "../include/option_pricer.h"
#include "../include/utils.h"
#include <cmath>
#include <random>
#include <iostream>
#include <vector>
#include <cstring>

// Forward declarations
std::vector<double> generate_gpm_path(double S, double r, double q, double sigma, double dt, int steps);
double back_propagation(std::vector<std::vector<double>>& paths, double K, double r, double q, double dt, int steps, bool is_call);
std::vector<double> polynomial_regression(const std::vector<double>& X, const std::vector<double>& Y);


double generate_random_normal(double mean, double sigma) {
    static std::random_device rd;  // Obtain a random number from hardware
    static std::mt19937 generator(rd()); // Seed the generator
    std::normal_distribution<double> distribution(mean, sigma); // Create a normal distribution

    return distribution(generator); // Generate a random number
}

double monte_carlo_asian_option(double S, double K, double r, double q, double sigma, double T, int steps, int num_simulations, bool is_call){
    double price_total = 0.0;
    double sum_squares = 0.0;
    double dt = T / steps;
    double drift = (r - q - sigma * sigma / 2.0) * dt;  // Risk-neutral drift adjusted for dividend yield
    double discount = std::exp(-r * T); // Discount factor for present value
    // number of simulations
    for (int i = 0; i < num_simulations; ++i){
        double simulated_price = S; // reset start price for each simulation
        double sum_path = 0.0;
        // number of steps per simulation
        for (int j = 0; j < steps; ++j) {
            double z = generate_random_normal(0, 1); // generate a standard normal random variable
            // simulate the price path using geometric Brownian motion
            simulated_price *= std::exp(drift + (sigma * sqrt(dt) * z));
            sum_path += simulated_price;
        }
        // Calculate average price along the path for Asian option
        double avg_price = sum_path / steps;
        double payoff = 0.0;
        if (is_call) {
            payoff = std::max(0.0, avg_price - K);
        } else {
            payoff = std::max(0.0, K - avg_price);
        }
        price_total += payoff;
        sum_squares += payoff * payoff; // accumulate squares for variance calculation
    } // Close the simulation loop
    // Calculate standard error
    double mean = price_total / num_simulations;
    double variance = (sum_squares / num_simulations) - (mean * mean);
    double std_error = sqrt(variance / num_simulations) * discount;
    double price = price_total / num_simulations * discount;
    std::cout << "Standard Error: " << std_error << std::endl;


    return price; // Return the average price discounted to present value
}


double monte_carlo_american_option(double S, double K, double r, double q, double sigma, double T, int steps, int num_simulations, bool is_call) {
    double dt = T / steps;
    std::vector<std::vector<double>> paths(num_simulations, std::vector<double>(steps + 1));
    for (int i = 0; i < paths.size(); ++i) {
        paths[i] = generate_gpm_path(S, r, q, sigma, dt, steps); // Pass dividend yield q into path generator
    }
    double v1_avg = back_propagation(paths, K, r, q, dt, steps, is_call);
    // Fix: Discount the average value at t=1 back to t=0 and take max with intrinsic at t=0
    double discount_last = std::exp(-r * dt);
    double continuation = v1_avg * discount_last;
    double intrinsic = is_call ? std::max(0.0, S - K) : std::max(0.0, K - S);
    return std::max(intrinsic, continuation);
}


std::vector<double> generate_gpm_path(double S, double r, double q, double sigma, double dt, int steps) {
    double simulated_price = S;
    double drift = (r - q - sigma * sigma / 2.0) * dt;  // Risk-neutral drift adjusted for dividend yield
    std::vector<double> path(steps + 1);
    path[0] = S; // Set the initial price
    for (int i = 0; i < steps; ++i) {
        double z = generate_random_normal(0,1); // generate a standard normal random variable
        simulated_price *= std::exp(drift + (sigma * sqrt(dt) * z));
        path[i + 1] = simulated_price; // Store the simulated price at each step
    }
    return path;
}


double back_propagation(std::vector<std::vector<double>>& paths, double K, double r, double q, double dt, int steps, bool is_call) {
    int M = paths.size();
    std::vector<double> cashflows(M, 0.0);

    // Initialize with terminal payoffs
    for (int i = 0; i < M; ++i) {
        double ST = paths[i][steps];
        cashflows[i] = is_call ? std::max(0.0, ST - K) : std::max(0.0, K - ST);
    }

    // Step backwards in time
    for (int t = steps-1; t > 0; --t) {
        std::vector<double> X;  // stock prices
        std::vector<double> Y;  // discounted future cashflows

        for (int i = 0; i < M; ++i) {
            double St = paths[i][t];
            double intrinsic = is_call ? std::max(0.0, St - K) : std::max(0.0, K - St);
            if (intrinsic > 0) {
                X.push_back(St);
                Y.push_back(cashflows[i] * std::exp(-r*dt)); // discounted future payoff
            }
        }

        // Only regress if we have enough ITM points
        if (X.size() > 3) {
            std::vector<double> coeffs = polynomial_regression(X, Y);

            // Update decisions path by path
            for (int i = 0; i < M; ++i) {
                double St = paths[i][t];
                double intrinsic = is_call ? std::max(0.0, St - K) : std::max(0.0, K - St);
                if (intrinsic > 0) {
                    double continuation = coeffs[0] + coeffs[1]*St + coeffs[2]*St*St;
                    if (intrinsic >= continuation) {
                        cashflows[i] = intrinsic; // exercise now
                    } else {
                        cashflows[i] = cashflows[i] * std::exp(-r*dt); // continue
                    }
                } else {
                    cashflows[i] = cashflows[i] * std::exp(-r*dt); // not ITM → continue
                }
            }
        } else {
            // Fallback: just discount if not enough points for regression
            for (int i = 0; i < M; ++i) {
                cashflows[i] = cashflows[i] * std::exp(-r*dt);
            }
        }
    }

    // At time 0: average over all discounted cashflows
    double price = 0.0;
    for (int i = 0; i < M; ++i) price += cashflows[i];
    return price / M;
}

// Solve linear regression: Y ≈ a0 + a1*X + a2*X^2
std::vector<double> polynomial_regression(const std::vector<double>& X, const std::vector<double>& Y) {
    int n = X.size();
    if (n == 0) return {0.0, 0.0, 0.0}; // avoid division by zero

    // Build sums for normal equations
    double Sx=0, Sx2=0, Sx3=0, Sx4=0;
    double Sy=0, Sxy=0, Sx2y=0;

    for (int i = 0; i < n; ++i) {
        double x = X[i];
        double y = Y[i];
        double x2 = x * x;

        Sx  += x;
        Sx2 += x2;
        Sx3 += x2 * x;
        Sx4 += x2 * x2;

        Sy  += y;
        Sxy += x * y;
        Sx2y += x2 * y;
    }

    // Normal equation system: 3x3 matrix
    // [ n,  Sx,  Sx2 ] [a0]   [ Sy  ]
    // [Sx, Sx2,  Sx3 ] [a1] = [ Sxy ]
    // [Sx2,Sx3,  Sx4 ] [a2]   [Sx2y]

    double A[3][3] = {{(double)n, Sx, Sx2},
                      {Sx, Sx2, Sx3},
                      {Sx2, Sx3, Sx4}};
    double b[3] = {Sy, Sxy, Sx2y};

    // Solve 3x3 system via Cramer's rule (simple, fine for small dimension)
    auto det3 = [](double M[3][3]) {
        return M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
             - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
             + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
    };

    double detA = det3(A);
    if (std::fabs(detA) < 1e-12) return {0.0, 0.0, 0.0};

    double A0[3][3], A1[3][3], A2[3][3];
    std::memcpy(A0, A, sizeof(A));
    std::memcpy(A1, A, sizeof(A));
    std::memcpy(A2, A, sizeof(A));
    for (int i=0;i<3;++i) { A0[i][0]=b[i]; A1[i][1]=b[i]; A2[i][2]=b[i]; }

    double a0 = det3(A0)/detA;
    double a1 = det3(A1)/detA;
    double a2 = det3(A2)/detA;

    return {a0, a1, a2};
}
