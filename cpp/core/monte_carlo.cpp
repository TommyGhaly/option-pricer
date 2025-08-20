#include "monte_carlo.h"
#include <cmath>
#include <random>
#include <iostream>


double generate_random_normal(double mean, double sigma) {
    static std::random_device rd;  // Obtain a random number from hardware
    static std::mt19937 generator(rd()); // Seed the generator
    std::normal_distribution<double> distribution(mean, sigma); // Create a normal distribution

    return distribution(generator); // Generate a random number
}

double monte_carlo_asian_option(double S, double K, double r, double sigma, double T, int steps, int num_simulations, bool is_call){
    double price_total = 0.0;
    double sum_squares = 0.0;
    double dt = T / steps;
    double drift = (r - sigma * sigma / 2.0) * dt;  // Risk-neutral drift
    double discount = std::exp(-r * T); // Discount factor for present value
    double simulated_price = S;

    // number of simulations
    for (int i = 0; i < num_simulations; ++i){
        simulated_price = S; // reset start price for each simulation
        // number of steps per simulation
        for (int j = 0; j < steps; ++j) {
            double z = generate_random_normal(0, 1); // generate a standard normal random variable
            // simulate the price path using geometric Brownian motion
            simulated_price *= std::exp(drift + (sigma * sqrt(dt) * z));
        }
        // accumlate the price for averaging
        double payoff = 0.0;
        if (is_call) {
            payoff = std::max(0.0, simulated_price - K);
        } else {
            payoff = std::max(0.0, K - simulated_price);
        price_total += payoff;
        sum_squares += payoff * payoff; // accumulate squares for variance calculation
    }
    // Calculate standard error
    double mean = price_total / num_simulations;
    double variance = (sum_squares / num_simulations) - (mean * mean);
    double std_error = sqrt(variance / num_simulations) * discount;
    double price = price_total / num_simulations * discount;
    std::cout << "Standard Error: " << std_error << std::endl;


    return price; // Return the average price discounted to present value
}
