#include <iostream>
#include <vector>
#include <fstream>
#include "network.h"
#include "metrics.h"

using namespace std;
int main(){
    // 1. Données d'entraînement (du projet)
    vector<double> x_train = {
        0.09291784, 0.46809093, 0.93089486, 0.67612654, 0.73441752,
        0.86847339, 0.49873225, 0.51083168, 0.18343972, 0.99380898,
        0.27840809, 0.38028817, 0.12055708, 0.56715537, 0.92005746,
        0.77072270, 0.85278176, 0.05315950, 0.87168699, 0.58858043
    };

    vector<int> y_train = {
        0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
        0, 1, 0, 1, 1, 0, 1, 0, 1, 1
    };

    // parametre du reseaux
    array<double,3> beta0 = {0.3,-1.0,-0.5};
    array<double,3> omega0 = {-1.0,1.8,0.65};
    double beta1 = 2.6;
    array<double,3> omega1 = {-24.0,-8.0,50};

    NeuralNetwork net(beta0,omega0,beta1,omega1);

    // teste avec le beta1 initial
    double L = computeLikelihood(net,x_train,y_train);
    double nll = computeNLL(net,x_train,y_train);
    std::cout << "Likelihood (beta1=" << beta1 << "): " << L << std::endl;
    std::cout << "NLL (beta1=" << beta1 << "): " << nll << std::endl;

    // recherche de l'optimum beta1
    ofstream file("result.csv");
    file << "beta1,likelihood,nll\n";

    double maxL=0,minNLL = 1e9;
    double bestBeta1_L = 0,bestBeta1_NLL=0;

    for (double b1=-10.0 ; b1 <=10.0;b1 += 0.05){
        net.setBeta1(b1);
        double L = computeLikelihood(net,x_train,y_train);
        double nll = computeNLL(net,x_train,y_train);

        file << b1 <<"," <<L << "," << nll <<"\n";

        if (L> maxL){
            maxL = L;
            bestBeta1_L = b1;
        }
        if (nll < minNLL){
            minNLL = nll;
            bestBeta1_NLL = b1;
        }
    }
    file.close();

    std::cout << "\n=== RÉSULTATS ===\n";
    std::cout << "Beta1 optimal (max L): " << bestBeta1_L << std::endl;
    std::cout << "Beta1 optimal (min NLL): " << bestBeta1_NLL << std::endl;

    return 0;
}
/*
 * gnuplot -p -e "set datafile separator ','; set multiplot layout 2,1; set title 'Likelihood'; plot 'result.csv' using 1:2 with lines lw 2; set title 'NLL'; plot 'result.csv' using 1:3 with lines lw 2 lc rgb 'red'; unset multiplot"

 */
