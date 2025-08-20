#include "binomial_tree.h"
#include "black_scholes.h"
#include <iostream>
#include <cassert>
#include <cmath>

void testValues() {
    double S = 100, K = 100, r = 0.05, sigma = 0.2, T = 1.0;

    // Test 1: Both models should give positive prices
    double bs_call = black_scholes(S, K, r, T, sigma, true);
    double bin_call = binomial_tree(S, K, r, sigma, T, 200, true);

    assert(bs_call > 0);
    assert(bin_call > 0);
    std::cout << "✓ Both models give positive call prices\n";

    // Test 2: Models should converge (within 1% for 200 steps)
    double difference = std::abs(bs_call - bin_call);
    double error_percent = (difference / bs_call) * 100;
    assert(error_percent < 1.0);
    std::cout << "✓ Models converge within 1% at 200 steps\n";

    // Test 3: Deep ITM call should be approximately S - K*exp(-rT)
    double deep_itm_bs = black_scholes(200, 100, r, T, sigma, true);
    double intrinsic = 200 - 100 * std::exp(-r * T);
    assert(std::abs(deep_itm_bs - intrinsic) < 0.5);
    std::cout << "✓ Deep ITM call approximates intrinsic value\n";

    std::cout << "\nAll validation tests passed!\n";
}

int main() {
    testValues();
    return 0;
}
