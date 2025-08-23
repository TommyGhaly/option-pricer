#ifndef BINOMIAL_TREE_H
#define BINOMIAL_TREE_H

double binomial_tree_euro(double S, double K, double r, double sigma, double T, int N, bool is_call);
double binomial_tree_american(double S, double K, double r, double sigma, double T, int N, bool is_call);


#endif
