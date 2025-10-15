#ifndef MODELS_H
#define MODELS_H
#include <vector>

// Add this struct definition
struct SurfacePoint {
    double K;
    double T;
    double iv;
};

double heston_model(double S, double K, double r, double q, double T, double v0, double kappa, double theta, double sigma, double rho, double lambda, bool is_call);
double jump_diffusion(double S, double K, double r, double q, double sigma, double lambda, double mu_j, double sigma_j, double T, bool is_call);


double european_local_vol_fdm(double S0, double K, double r, double q, double T,
                              const std::vector<SurfacePoint>& iv_surface,
                              bool is_call,
                              int N_S = 200, int N_T = 100);

double american_local_vol_fdm(double S0, double K, double r, double q, double T,
                              const std::vector<SurfacePoint>& iv_surface,
                              bool is_call,
                              int N_S = 200, int N_T = 100);


double SABRImpliedVol(double F, double K, double T, double alpha, double beta, double rho, double nu);
double SABROptionPrice(double S, double K, double r, double T, double F,
                      double alpha, double beta, double rho, double nu, bool isCall);
void CalibratesSABR(const double* strikes, const double* marketVols, int n,
                   double F, double T, double fixedBeta,
                   double& alpha, double& rho, double& nu);
double SABRDelta(double S, double K, double r, double T, double F,
                double alpha, double beta, double rho, double nu, bool isCall);
double SABRGamma(double S, double K, double r, double T, double F,
                double alpha, double beta, double rho, double nu, bool isCall);
double SABRVega(double S, double K, double r, double T, double F,
               double alpha, double beta, double rho, double nu, bool isCall);
double SABRVolga(double S, double K, double r, double T, double F,
                double alpha, double beta, double rho, double nu);
double SABRVanna(double S, double K, double r, double T, double F,
                double alpha, double beta, double rho, double nu);

#endif // MODELS_H
