#ifndef NETWORK_H
#define NETWORK_H

#include <array>
#include <vector>
#include <cmath>

using namespace std;

class NeuralNetwork{
    private:
        array<double,3> beta0;
        array<double,3> omega0;
        double beta1;
        array<double,3> omega1;

    public:
    NeuralNetwork(const array<double,3>& b0,const array<double,3>& w0,
        double b1,const array<double,3>& w1);

    // activation functions
    double relu(double z) const;
    double sigmoid(double z) const;

    // Propagation functions
    array<double,3> computeHidden(double x) const;
    double computeOutput(const array<double,3>& hidden) const;
    double predictProbability(double x) const;

    // beta setter
    void setBeta1(double newBeta1);
};

#endif
