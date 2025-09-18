#include "../include/models.h"
#include <cmath>
#include <algorithm>

double SABRImpliedVol(double F, double K, double T, double alpha, double beta, double rho, double nu) {
    const double EPSILON = 1e-10;

    // ATM case
    if (std::abs(F - K) < EPSILON) {
        double f_beta = std::pow(F, 1.0 - beta);
        double term1 = alpha / f_beta;

        double A = (2.0 - 3.0 * rho * rho) * nu * nu;
        double B = rho * beta * nu * alpha / f_beta;
        double C = beta * (beta - 2.0) * alpha * alpha / (f_beta * f_beta);

        return term1 * (1.0 + T / 24.0 * (A + 4.0 * B + C));
    }

    // OTM/ITM case
    double FK = std::sqrt(F * K);
    double FK_beta = std::pow(FK, 1.0 - beta);
    double log_FK = std::log(F / K);
    double log_FK2 = log_FK * log_FK;

    // Calculate z
    double z = (nu / alpha) * FK_beta * log_FK;

    // Calculate x(z) with stability check
    double x;
    if (std::abs(z) < EPSILON) {
        x = z * (1.0 - 0.5 * rho * z);
    } else {
        double sqrt_term = std::sqrt(1.0 - 2.0 * rho * z + z * z);
        x = std::log((sqrt_term + z - rho) / (1.0 - rho));
    }

    // First factor
    double denominator = FK_beta * (1.0 + log_FK2 / 24.0 + log_FK2 * log_FK2 / 1920.0);
    double factor1 = alpha / denominator;

    // z/x ratio
    double z_over_x = std::abs(x) < EPSILON ? 1.0 : z / x;

    // Time correction terms
    double A = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
    double B = rho * beta * nu * alpha / (4.0 * FK_beta);
    double C = beta * (2.0 - beta) * alpha * alpha / (24.0 * FK_beta * FK_beta);

    double factor3 = 1.0 + T * (A + B + C);

    return factor1 * z_over_x * factor3;
}

double SABROptionPrice(double S, double K, double r, double T, double F,
                      double alpha, double beta, double rho, double nu, bool isCall) {
    // Get SABR implied volatility
    double sigma = SABRImpliedVol(F, K, T, alpha, beta, rho, nu);

    // Black formula
    double d1 = (std::log(F / K) + 0.5 * sigma * sigma * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    double Nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
    double Nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));

    double price;
    if (isCall) {
        price = std::exp(-r * T) * (F * Nd1 - K * Nd2);
    } else {
        double Nmd1 = 1.0 - Nd1;
        double Nmd2 = 1.0 - Nd2;
        price = std::exp(-r * T) * (K * Nmd2 - F * Nmd1);
    }

    return price;
}

void CalibratesSABR(const double* strikes, const double* marketVols, int n,
                   double F, double T, double fixedBeta,
                   double& alpha, double& rho, double& nu) {
    // Initial parameter guesses
    double atmVol = marketVols[n/2]; // Assume middle strike is near ATM
    alpha = atmVol * std::pow(F, 1.0 - fixedBeta);
    rho = -0.3;
    nu = 0.4;

    // Optimization parameters
    const int maxIter = 500;
    const double tol = 1e-6;
    double learningRate = 0.01;

    for (int iter = 0; iter < maxIter; ++iter) {
        // Calculate current error
        double error = 0.0;
        for (int i = 0; i < n; ++i) {
            double modelVol = SABRImpliedVol(F, strikes[i], T, alpha, fixedBeta, rho, nu);
            error += std::pow(modelVol - marketVols[i], 2);
        }

        // Numerical gradients
        const double h = 1e-5;

        // Alpha gradient
        double errorAlphaPlus = 0.0;
        for (int i = 0; i < n; ++i) {
            double vol = SABRImpliedVol(F, strikes[i], T, alpha + h, fixedBeta, rho, nu);
            errorAlphaPlus += std::pow(vol - marketVols[i], 2);
        }
        double gradAlpha = (errorAlphaPlus - error) / h;

        // Rho gradient
        double errorRhoPlus = 0.0;
        for (int i = 0; i < n; ++i) {
            double vol = SABRImpliedVol(F, strikes[i], T, alpha, fixedBeta, rho + h, nu);
            errorRhoPlus += std::pow(vol - marketVols[i], 2);
        }
        double gradRho = (errorRhoPlus - error) / h;

        // Nu gradient
        double errorNuPlus = 0.0;
        for (int i = 0; i < n; ++i) {
            double vol = SABRImpliedVol(F, strikes[i], T, alpha, fixedBeta, rho, nu + h);
            errorNuPlus += std::pow(vol - marketVols[i], 2);
        }
        double gradNu = (errorNuPlus - error) / h;

        // Update parameters
        alpha -= learningRate * gradAlpha;
        rho -= learningRate * gradRho;
        nu -= learningRate * gradNu;

        // Enforce constraints
        alpha = std::max(0.001, alpha);
        rho = std::max(-0.999, std::min(0.999, rho));
        nu = std::max(0.001, nu);

        // Check convergence
        if (std::abs(gradAlpha) < tol && std::abs(gradRho) < tol && std::abs(gradNu) < tol) {
            break;
        }

        // Adaptive learning rate
        if (iter % 50 == 0) {
            learningRate *= 0.9;
        }
    }
}

double SABRDelta(double S, double K, double r, double T, double F,
                double alpha, double beta, double rho, double nu, bool isCall) {
    const double h = 0.01 * S;

    double F_up = (S + h) * std::exp(r * T);
    double F_down = (S - h) * std::exp(r * T);

    double price_up = SABROptionPrice(S + h, K, r, T, F_up, alpha, beta, rho, nu, isCall);
    double price_down = SABROptionPrice(S - h, K, r, T, F_down, alpha, beta, rho, nu, isCall);

    return (price_up - price_down) / (2.0 * h);
}

double SABRGamma(double S, double K, double r, double T, double F,
                double alpha, double beta, double rho, double nu, bool isCall) {
    const double h = 0.01 * S;

    double F_up = (S + h) * std::exp(r * T);
    double F_down = (S - h) * std::exp(r * T);

    double price_up = SABROptionPrice(S + h, K, r, T, F_up, alpha, beta, rho, nu, isCall);
    double price_center = SABROptionPrice(S, K, r, T, F, alpha, beta, rho, nu, isCall);
    double price_down = SABROptionPrice(S - h, K, r, T, F_down, alpha, beta, rho, nu, isCall);

    return (price_up - 2.0 * price_center + price_down) / (h * h);
}

double SABRVega(double S, double K, double r, double T, double F,
               double alpha, double beta, double rho, double nu, bool isCall) {
    const double h = 0.01 * alpha;

    double price_up = SABROptionPrice(S, K, r, T, F, alpha + h, beta, rho, nu, isCall);
    double price_down = SABROptionPrice(S, K, r, T, F, alpha - h, beta, rho, nu, isCall);

    return (price_up - price_down) / (2.0 * h) * alpha; // Scale by alpha for percentage vega
}

double SABRVolga(double S, double K, double r, double T, double F,
                double alpha, double beta, double rho, double nu) {
    const double h = 0.001;

    double vol_base = SABRImpliedVol(F, K, T, alpha, beta, rho, nu);
    double vol_up = SABRImpliedVol(F * (1 + h), K, T, alpha, beta, rho, nu);
    double vol_down = SABRImpliedVol(F * (1 - h), K, T, alpha, beta, rho, nu);

    double dVol_dF = (vol_up - vol_down) / (2 * h * F);

    // Second derivative approximation
    double d2Vol_dF2 = (vol_up - 2 * vol_base + vol_down) / (h * h * F * F);

    return S * S * d2Vol_dF2;
}

double SABRVanna(double S, double K, double r, double T, double F,
                double alpha, double beta, double rho, double nu) {
    const double h_S = 0.01 * S;
    const double h_vol = 0.01;

    // Calculate cross-derivative of price w.r.t spot and vol
    double vol_base = SABRImpliedVol(F, K, T, alpha, beta, rho, nu);
    double vol_up = SABRImpliedVol(F * (1 + h_S/S), K, T, alpha, beta, rho, nu);

    double dVol_dS = (vol_up - vol_base) / h_S;

    // Vanna is approximately S * dVol/dS
    return S * dVol_dS;
}
