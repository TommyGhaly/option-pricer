#include "binomial_tree.h"
#include "black_scholes.h"
#include "greeks.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

struct TestCase {
    std::string name;
    double S;      // Spot price
    double K;      // Strike price
    double r;      // Risk-free rate
    double sigma;  // Volatility
    double T;      // Time to maturity
    bool is_call;  // Call or Put
};

void runComparison(const TestCase& test) {
    std::cout << "\n========================================\n";
    std::cout << test.name << "\n";
    std::cout << "Parameters: S=" << test.S << ", K=" << test.K
              << ", r=" << test.r << ", σ=" << test.sigma
              << ", T=" << test.T << ", Type=" << (test.is_call ? "Call" : "Put") << "\n";
    std::cout << "----------------------------------------\n";

    // Calculate Black-Scholes price
    double bs_price = black_scholes(test.S, test.K, test.r, test.T, test.sigma, test.is_call);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Black-Scholes Price: " << bs_price << "\n\n";

    // Test binomial tree with different step sizes
    std::cout << "Steps\tBinomial Price\tDifference\tError %\n";
    std::cout << "-----\t--------------\t----------\t-------\n";

    std::vector<int> steps = {10, 25, 50, 100, 200, 500, 1000};

    for (int N : steps) {
        double bin_price = binomial_tree(test.S, test.K, test.r, test.sigma, test.T, N, test.is_call);
        double diff = bin_price - bs_price;
        double error_pct = (std::abs(diff) / bs_price) * 100;

        std::cout << N << "\t" << bin_price << "\t"
                  << std::showpos << diff << "\t"
                  << std::noshowpos << error_pct << "%\n";
    }
}

void runPutCallParityTest() {
    std::cout << "\n========================================\n";
    std::cout << "PUT-CALL PARITY TEST\n";
    std::cout << "C - P = S - K*exp(-rT)\n";
    std::cout << "========================================\n";

    double S = 100, K = 100, r = 0.05, sigma = 0.2, T = 1.0;

    // Black-Scholes
    double bs_call = black_scholes(S, K, r, T, sigma, true);
    double bs_put = black_scholes(S, K, r, T, sigma, false);
    double bs_parity = bs_call - bs_put;
    double theoretical_parity = S - K * std::exp(-r * T);

    std::cout << "Black-Scholes:\n";
    std::cout << "Call: " << bs_call << ", Put: " << bs_put << "\n";
    std::cout << "C - P = " << bs_parity << "\n";
    std::cout << "S - K*exp(-rT) = " << theoretical_parity << "\n";
    std::cout << "Difference: " << std::abs(bs_parity - theoretical_parity) << "\n\n";

    // Binomial with 500 steps
    double bin_call = binomial_tree(S, K, r, sigma, T, 500, true);
    double bin_put = binomial_tree(S, K, r, sigma, T, 500, false);
    double bin_parity = bin_call - bin_put;

    std::cout << "Binomial (500 steps):\n";
    std::cout << "Call: " << bin_call << ", Put: " << bin_put << "\n";
    std::cout << "C - P = " << bin_parity << "\n";
    std::cout << "Difference from theoretical: " << std::abs(bin_parity - theoretical_parity) << "\n";
}

void runFullGreeksTest() {
    std::cout << "\n========================================\n";
    std::cout << "FULL GREEKS COMPARISON TEST\n";
    std::cout << "Analytical vs Finite Difference\n";
    std::cout << "========================================\n";

    double S = 100, K = 100, r = 0.05, sigma = 0.2, T = 0.25;

    // Small changes for finite differences
    double dS = 0.01;      // For delta and gamma
    double dSigma = 0.001; // For vega
    double dT = 0.001;     // For theta (1 day ~ 1/365)
    double dR = 0.0001;    // For rho

    std::cout << std::fixed << std::setprecision(6);

    // Test both call and put options
    for (bool is_call : {true, false}) {
        std::cout << "\n" << (is_call ? "CALL" : "PUT") << " OPTION GREEKS:\n";
        std::cout << "----------------------------------------\n";

        // Current price
        double price = black_scholes(S, K, r, T, sigma, is_call);
        std::cout << "Option Price: " << price << "\n\n";

        // DELTA - first derivative w.r.t. S
        double price_up = black_scholes(S + dS, K, r, T, sigma, is_call);
        double price_down = black_scholes(S - dS, K, r, T, sigma, is_call);
        double fd_delta = (price_up - price_down) / (2 * dS);
        double analytical_delta = delta(S, K, r, T, sigma, is_call);

        std::cout << "Delta:\n";
        std::cout << "  Analytical:         " << analytical_delta << "\n";
        std::cout << "  Finite Difference:  " << fd_delta << "\n";
        std::cout << "  Absolute Error:     " << std::abs(analytical_delta - fd_delta) << "\n\n";

        // GAMMA - second derivative w.r.t. S
        double fd_gamma = (price_up - 2 * price + price_down) / (dS * dS);
        double analytical_gamma = gamma(S, K, r, T, sigma);

        std::cout << "Gamma:\n";
        std::cout << "  Analytical:         " << analytical_gamma << "\n";
        std::cout << "  Finite Difference:  " << fd_gamma << "\n";
        std::cout << "  Absolute Error:     " << std::abs(analytical_gamma - fd_gamma) << "\n\n";

        // THETA - derivative w.r.t. time (negative because we decrease T)
        double price_t_down = black_scholes(S, K, r, T - dT, sigma, is_call);
        double fd_theta = -(price - price_t_down) / dT;  // Negative because option loses value as time passes
        double analytical_theta = theta(S, K, r, T, sigma, is_call);

        std::cout << "Theta:\n";
        std::cout << "  Analytical:         " << analytical_theta << "\n";
        std::cout << "  Finite Difference:  " << fd_theta << "\n";
        std::cout << "  Absolute Error:     " << std::abs(analytical_theta - fd_theta) << "\n\n";

        // VEGA - derivative w.r.t. volatility
        double price_sigma_up = black_scholes(S, K, r, T, sigma + dSigma, is_call);
        double price_sigma_down = black_scholes(S, K, r, T, sigma - dSigma, is_call);
        double fd_vega = (price_sigma_up - price_sigma_down) / (2 * dSigma);
        double analytical_vega = vega(S, K, r, T, sigma);

        std::cout << "Vega:\n";
        std::cout << "  Analytical:         " << analytical_vega << "\n";
        std::cout << "  Finite Difference:  " << fd_vega << "\n";
        std::cout << "  Absolute Error:     " << std::abs(analytical_vega - fd_vega) << "\n\n";

        // RHO - derivative w.r.t. interest rate
        double price_r_up = black_scholes(S, K, r + dR, T, sigma, is_call);
        double price_r_down = black_scholes(S, K, r - dR, T, sigma, is_call);
        double fd_rho = (price_r_up - price_r_down) / (2 * dR);
        double analytical_rho = rho(S, K, r, T, sigma, is_call);

        std::cout << "Rho:\n";
        std::cout << "  Analytical:         " << analytical_rho << "\n";
        std::cout << "  Finite Difference:  " << fd_rho << "\n";
        std::cout << "  Absolute Error:     " << std::abs(analytical_rho - fd_rho) << "\n";
    }
}

void runGreeksStressTest() {
    std::cout << "\n========================================\n";
    std::cout << "GREEKS STRESS TEST\n";
    std::cout << "Testing Greeks under extreme conditions\n";
    std::cout << "========================================\n";

    struct GreeksTestCase {
        std::string name;
        double S, K, r, sigma, T;
    };

    std::vector<GreeksTestCase> test_cases = {
        {"Deep ITM Call", 150, 100, 0.05, 0.2, 0.5},
        {"Deep OTM Put", 50, 100, 0.05, 0.2, 0.5},
        {"Near Expiry", 100, 100, 0.05, 0.2, 0.01},
        {"High Volatility", 100, 100, 0.05, 0.8, 0.25},
        {"Zero Interest", 100, 100, 0.0, 0.2, 0.5}
    };

    std::cout << std::fixed << std::setprecision(6);

    for (const auto& test : test_cases) {
        std::cout << "\n" << test.name << ":\n";
        std::cout << "S=" << test.S << ", K=" << test.K << ", r=" << test.r
                  << ", σ=" << test.sigma << ", T=" << test.T << "\n";

        // Calculate all Greeks for call option
        double call_delta = delta(test.S, test.K, test.r, test.T, test.sigma, true);
        double call_gamma = gamma(test.S, test.K, test.r, test.T, test.sigma);
        double call_theta = theta(test.S, test.K, test.r, test.T, test.sigma, true);
        double call_vega = vega(test.S, test.K, test.r, test.T, test.sigma);
        double call_rho = rho(test.S, test.K, test.r, test.T, test.sigma, true);

        std::cout << "Call Greeks: Δ=" << call_delta << ", Γ=" << call_gamma
                  << ", Θ=" << call_theta << ", ν=" << call_vega
                  << ", ρ=" << call_rho << "\n";
    }
}

void runGreeksRelationshipTest() {
    std::cout << "\n========================================\n";
    std::cout << "GREEKS RELATIONSHIPS TEST\n";
    std::cout << "Testing known Greek relationships\n";
    std::cout << "========================================\n";

    double S = 100, K = 100, r = 0.05, sigma = 0.2, T = 0.5;

    // Test 1: Put-Call Delta relationship
    double call_delta = delta(S, K, r, T, sigma, true);
    double put_delta = delta(S, K, r, T, sigma, false);
    double delta_diff = call_delta - put_delta;

    std::cout << "1. Put-Call Delta Relationship (Call_Δ - Put_Δ = 1):\n";
    std::cout << "   Call Delta: " << call_delta << "\n";
    std::cout << "   Put Delta:  " << put_delta << "\n";
    std::cout << "   Difference: " << delta_diff << " (should be 1.0)\n";
    std::cout << "   Error:      " << std::abs(delta_diff - 1.0) << "\n\n";

    // Test 2: Gamma is same for call and put
    double call_gamma = gamma(S, K, r, T, sigma);
    double put_gamma = gamma(S, K, r, T, sigma);  // Should be identical

    std::cout << "2. Call and Put Gamma Equality:\n";
    std::cout << "   Call Gamma: " << call_gamma << "\n";
    std::cout << "   Put Gamma:  " << put_gamma << "\n";
    std::cout << "   Difference: " << std::abs(call_gamma - put_gamma) << " (should be 0)\n\n";

    // Test 3: Vega is same for call and put
    double call_vega = vega(S, K, r, T, sigma);
    double put_vega = vega(S, K, r, T, sigma);  // Should be identical

    std::cout << "3. Call and Put Vega Equality:\n";
    std::cout << "   Call Vega: " << call_vega << "\n";
    std::cout << "   Put Vega:  " << put_vega << "\n";
    std::cout << "   Difference: " << std::abs(call_vega - put_vega) << " (should be 0)\n";
}

int main() {
    std::cout << "OPTIONS PRICING MODEL COMPARISON TEST\n";
    std::cout << "Comparing Black-Scholes vs Binomial Tree\n";

    // Define test cases
    std::vector<TestCase> tests = {
        {"ATM Call Option", 100, 100, 0.05, 0.2, 1.0, true},
        {"ATM Put Option", 100, 100, 0.05, 0.2, 1.0, false},
        {"ITM Call Option", 110, 100, 0.05, 0.2, 1.0, true},
        {"OTM Put Option", 110, 100, 0.05, 0.2, 1.0, false},
        {"High Volatility Call", 100, 100, 0.05, 0.4, 1.0, true},
        {"Short Maturity Call", 100, 100, 0.05, 0.2, 0.0833, true}, // 1 month
        {"Deep ITM Call", 150, 100, 0.05, 0.2, 1.0, true},
        {"Deep OTM Put", 50, 100, 0.05, 0.2, 1.0, false}
    };

    // Run comparison tests
    for (const auto& test : tests) {
        runComparison(test);
    }

    // Additional tests
    runPutCallParityTest();
    runFullGreeksTest();        // Comprehensive Greeks testing
    runGreeksStressTest();      // Greeks under extreme conditions
    runGreeksRelationshipTest(); // Test known Greek relationships

    std::cout << "\n========================================\n";
    std::cout << "All tests completed!\n";

    return 0;
}
