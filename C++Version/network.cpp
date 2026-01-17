#include "network.h"
#include <cmath>

NeuralNetwork::NeuralNetwork(const array<double,3>& b0,
                             const array<double,3>& w0,
                             double b1,
                             const array<double,3>& w1)
    : beta0(b0), omega0(w0), beta1(b1), omega1(w1) {}

double NeuralNetwork::relu(double z) const{
    return max(0.0,z);
}

double NeuralNetwork::sigmoid(double z) const{
    if (z > 500) z = 500;
    if (z < -500) z = -500;
    return 1.0/(1.0+exp(-z));
}

array<double,3> NeuralNetwork::computeHidden(double x)const{
    array<double,3> h;
    for (int i =0;i<3;i++){
    double z = omega0[i]*x + beta0[i];
    h[i] = relu(z);
    }
    return h;
}
double NeuralNetwork::computeOutput(const array<double,3>& h)const{
    double f = beta1;
    for (int i = 0;i < 3 ; i++){
        f += omega1[i] * h[i];
    }
    return f;
}

double NeuralNetwork::predictProbability(double x) const{
    auto h = computeHidden(x);
    double f = computeOutput(h);
    return sigmoid(f);
}

void NeuralNetwork::setBeta1(double newBeta1) {
    beta1 = newBeta1;
}
