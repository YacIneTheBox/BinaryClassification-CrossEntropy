#include "metrics.h"
#include <cmath>       // ✓ OK mais devrait être <cmath> pas "cmath"
#include <algorithm>   // ❌ MANQUE pour min/max
#include <vector>


double computeLikelihood(const NeuralNetwork& net,
                         const std::vector<double>& x_train,
                         const std::vector<int>& y_train) {
    double L = 1.0;
    for (size_t i = 0; i < x_train.size(); i++) {
        double p = net.predictProbability(x_train[i]);
        if (y_train[i] == 1) {
            L *= p;
        } else {
            L *= (1.0 - p);
        }
    }
    return L;
}

double computeNLL(const NeuralNetwork &net,
    const vector<double> &x_train,
    const vector<int> &y_train){
        double nll = 0.0;
        const double epsilon = 1e-15;

        for (size_t i = 0; i < x_train.size(); i++) {
            double p = net.predictProbability(x_train[i]);

            // CLip eviter log
            p = max(epsilon,min(1.0 - epsilon,p));

            // BCE
            nll -= y_train[i] * log(p) + (1.0 - y_train[i]) * log(1.0 - p);
        }
        return nll;
    }
