#ifndef METRICS_H
#define METRICS_H

#include "network.h"
#include <vector>

double computeLikelihood(const NeuralNetwork& net,
const vector<double>& x_train,
const vector<int>& y_train);

double computeNLL(const NeuralNetwork& net,
const vector<double>& x_train,
const vector<int>& y_train);

#endif
