#include "binomial_tree.h"
#include "black_scholes.h"
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
    double bs_price = calculate_option_price(test.S, test.K, test.r, test.T, test.sigma, test.is_call);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Black-Scholes Price: " << bs_price << "\n\n";

    // Test binomial tree with different step sizes
    std::cout << "Steps\tBinomial Price\tDifference\tError %\n";
    std::cout << "-----\t--------------\t----------\t-------\n";

    std::vector<int> steps = {10, 25, 50, 100, 200, 500, 1000};

    for (int N : steps) {
        double bin_price = priceOptionBinTree(test.S, test.K, test.r, test.sigma, test.T, N, test.is_call);
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
    double bs_call = calculate_option_price(S, K, r, T, sigma, true);
    double bs_put = calculate_option_price(S, K, r, T, sigma, false);
    double bs_parity = bs_call - bs_put;
    double theoretical_parity = S - K * std::exp(-r * T);

    std::cout << "Black-Scholes:\n";
    std::cout << "Call: " << bs_call << ", Put: " << bs_put << "\n";
    std::cout << "C - P = " << bs_parity << "\n";
    std::cout << "S - K*exp(-rT) = " << theoretical_parity << "\n";
    std::cout << "Difference: " << std::abs(bs_parity - theoretical_parity) << "\n\n";

    // Binomial with 500 steps
    double bin_call = priceOptionBinTree(S, K, r, sigma, T, 500, true);
    double bin_put = priceOptionBinTree(S, K, r, sigma, T, 500, false);
    double bin_parity = bin_call - bin_put;

    std::cout << "Binomial (500 steps):\n";
    std::cout << "Call: " << bin_call << ", Put: " << bin_put << "\n";
    std::cout << "C - P = " << bin_parity << "\n";
    std::cout << "Difference from theoretical: " << std::abs(bin_parity - theoretical_parity) << "\n";
}

void runGreeksApproximation() {
    std::cout << "\n========================================\n";
    std::cout << "GREEKS APPROXIMATION TEST (DELTA)\n";
    std::cout << "========================================\n";

    double S = 100, K = 100, r = 0.05, sigma = 0.2, T = 0.25;
    double dS = 0.01; // Small change in stock price

    // Calculate Delta using finite differences
    double price_up = calculate_option_price(S + dS, K, r, T, sigma, true);
    double price_down = calculate_option_price(S - dS, K, r, T, sigma, true);
    double bs_delta = (price_up - price_down) / (2 * dS);

    double bin_up = priceOptionBinTree(S + dS, K, r, sigma, T, 500, true);
    double bin_down = priceOptionBinTree(S - dS, K, r, sigma, T, 500, true);
    double bin_delta = (bin_up - bin_down) / (2 * dS);

    std::cout << "Call Option Delta:\n";
    std::cout << "Black-Scholes: " << bs_delta << "\n";
    std::cout << "Binomial Tree: " << bin_delta << "\n";
    std::cout << "Difference: " << std::abs(bs_delta - bin_delta) << "\n";
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
    runGreeksApproximation();

    std::cout << "\n========================================\n";
    std::cout << "All tests completed!\n";

    return 0;
}
